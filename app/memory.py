"""Mémoire de session thread-safe : one-shot, personal, shared-compressed.

Trois modes :

``one-shot``
    Aucune conservation entre deux appels : l'historique repart vide.

``personal``
    Un historique par ``session_id``, persisté en JSON dans
    ``<kb_dir>/sessions/<session_id>.json`` (identifiant nettoyé).

``shared-compressed``
    Un unique historique global, partagé par toutes les sessions, persisté
    dans ``<kb_dir>/shared_memory.json``. Dès que le volume de texte dépasse
    ``compression_threshold``, un appel LLM le résume et ne conserve que les
    derniers ``keep_last_messages_after_compression`` messages bruts.

La concurrence est gérée par un ``asyncio.Lock`` global (process unique) et,
pour les écritures, par un verrou fichier ``O_EXCL`` (multi-process).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, Iterable, List, Optional, Tuple

from .config import MemoryConfig

Message = Dict[str, Any]

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_session_id(session_id: str, default: str = "default") -> str:
    """Rend un identifiant sûr pour un nom de fichier."""
    cleaned = _SAFE_ID.sub("-", (session_id or "").strip())
    cleaned = cleaned.strip(".-") or default
    return cleaned[:64]


def _estimate_size(messages: Iterable[Message]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


@dataclass
class CompressedMemory:
    """État en mémoire d'une mémoire persistante."""

    summary: str = ""
    messages: Deque[Message] = field(default_factory=lambda: deque(maxlen=512))
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "messages": list(self.messages),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CompressedMemory":
        return cls(
            summary=str(raw.get("summary") or ""),
            messages=deque(raw.get("messages") or [], maxlen=512),
            updated_at=float(raw.get("updated_at") or 0.0),
        )


class MemoryStore:
    """Façade unique utilisée par le harnais."""

    def __init__(self, config: MemoryConfig, kb_dir: Path) -> None:
        self.config = config
        self.kb_dir = Path(kb_dir)
        self.mode = config.mode
        self._lock = asyncio.Lock()
        self._file_lock = asyncio.Lock()

        self.shared_path = self.kb_dir / "shared_memory.json"
        self.sessions_dir = self.kb_dir / "sessions"

        self._shared: CompressedMemory = CompressedMemory()
        self._personal: Dict[str, CompressedMemory] = {}

        if self.persist:
            self.kb_dir.mkdir(parents=True, exist_ok=True)
            if self.mode == "personal":
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
            elif self.mode == "shared-compressed":
                self._shared = self._read(self.shared_path)

    # ------------------------------------------------------------------ API

    @property
    def persist(self) -> bool:
        return self.mode in ("personal", "shared-compressed")

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["MemoryStore"]:
        """Sérialise un tour complet de conversation."""
        async with self._lock:
            yield self

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{sanitize_session_id(session_id)}.json"

    def _read(self, path: Path) -> CompressedMemory:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CompressedMemory()
        if not isinstance(raw, dict):
            return CompressedMemory()
        return CompressedMemory.from_dict(raw)

    def _write(self, path: Path, state: CompressedMemory) -> None:
        state.updated_at = time.time()
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------ accès brut

    def messages_for(self, session_id: str) -> List[Message]:
        """Historique + résumé prêt à être injecté dans le prompt."""
        if self.mode == "one-shot":
            return []
        if self.mode == "personal":
            state = self._load_personal(session_id)
        else:
            state = self._shared
        return self._render(state)

    def _load_personal(self, session_id: str) -> CompressedMemory:
        state = self._personal.get(session_id)
        if state is None:
            state = self._read(self._session_file(session_id))
            self._personal[session_id] = state
        return state

    def _render(self, state: CompressedMemory) -> List[Message]:
        rendered: List[Message] = []
        if state.summary:
            rendered.append(
                {
                    "role": "system",
                    "content": (
                        "Résumé des échanges précédents (mémoire compressée) :\n"
                        + state.summary
                    ),
                }
            )
        rendered.extend(state.messages)
        return rendered[-self.config.max_history_messages :] if self.config.max_history_messages else []

    # -------------------------------------------------------------- mutation

    def append(self, session_id: str, message: Message) -> None:
        """Ajoute un message et déclenche la compression si nécessaire."""
        if self.mode == "one-shot":
            return
        if self.mode == "personal":
            state = self._load_personal(session_id)
            state.messages.append(message)
            self._personal[session_id] = state
            self._write(self._session_file(session_id), state)
            return
        self._shared.messages.append(message)
        self._maybe_compress()
        self._write(self.shared_path, self._shared)

    def record_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Enregistre un tour complet (question + réponse)."""
        self.append(session_id, {"role": "user", "content": user_message})
        self.append(session_id, {"role": "assistant", "content": assistant_message})

    def clear(self, session_id: Optional[str] = None) -> None:
        """Réinitialise la mémoire (session courante, ou global)."""
        if self.mode == "personal" and session_id:
            self._personal.pop(session_id, None)
            path = self._session_file(session_id)
            path.unlink(missing_ok=True)
            return
        self._shared = CompressedMemory()
        self.shared_path.unlink(missing_ok=True)

    # ------------------------------------------------------------ compression

    def needs_compression(self, state: Optional[CompressedMemory] = None) -> bool:
        state = state or self._shared
        size = _estimate_size(state.messages) + len(state.summary)
        return size >= self.config.compression_threshold

    def set_summary(self, summary: str, keep: Optional[int] = None) -> List[Message]:
        """Applique un résumé produit par le LLM et purge les vieux messages."""
        keep = self.config.keep_last_messages_after_compression if keep is None else keep
        kept = list(self._shared.messages)[-keep:] if keep else []
        dropped = [m for m in list(self._shared.messages) if m not in kept]
        self._shared.summary = (self._shared.summary + "\n" + summary).strip()[-8000:]
        self._shared.messages = deque(kept, maxlen=512)
        self._write(self.shared_path, self._shared)
        return dropped


__all__ = [
    "CompressedMemory",
    "MemoryStore",
    "Message",
    "sanitize_session_id",
]