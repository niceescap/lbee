"""Chargement dynamique des outils et boucle agentique OpenAI-compatible."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

from .config import LbeeConfig, api_key, system_prompt
from .memory import MemoryStore, Message, sanitize_session_id

log = logging.getLogger("lbee.harness")

ToolFn = Callable[..., Any]

TOOL_ERROR_PREFIX = "TOOL_ERROR"


class ToolNotFound(RuntimeError):
    """Outil demandé par le modèle mais absent du répertoire class-tools."""


class ToolExecutionError(RuntimeError):
    """L'outil a levé une exception pendant son exécution."""


# --------------------------------------------------------------- chargement


def _load_module(path: Path) -> Optional[ModuleType]:
    name = f"lbee_tools.{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        log.warning("module non chargeable: %s", path)
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # un outil cassé ne doit pas tuer le service
        log.exception("échec de chargement de %s: %s", path.name, exc)
        sys.modules.pop(name, None)
        return None
    return module


def discover_tools(tools_dir: Path) -> Dict[str, Tuple[ToolFn, Dict[str, Any]]]:
    """Charge ``SCHEMA`` + ``execute`` de chaque module de ``tools_dir``."""
    tools: Dict[str, Tuple[ToolFn, Dict[str, Any]]] = {}
    if not tools_dir.is_dir():
        log.warning("tools_dir introuvable: %s", tools_dir)
        return tools

    for path in sorted(tools_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = _load_module(path)
        if module is None:
            continue

        schema = getattr(module, "SCHEMA", None)
        func = getattr(module, "execute", None)
        if schema is None or not callable(func):
            log.warning("%s: SCHEMA ou execute() manquant, outil ignoré", path.name)
            continue

        try:
            name = schema["function"]["name"]
        except (KeyError, TypeError):
            log.warning("%s: SCHEMA malformé (function.name attendu)", path.name)
            continue

        if name in tools:
            log.warning("nom d'outil dupliqué: %s", name)
        tools[name] = (func, schema)
        log.info("outil chargé: %s (%s)", name, path.name)

    return tools


def openai_schemas(tools: Dict[str, Tuple[ToolFn, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [schema for _, schema in tools.values()]


# ---------------------------------------------------------------- exécution


def _coerce_arguments(func: ToolFn, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Filtre les arguments selon la signature réelle de ``execute()``."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # builtins / C callables
        return arguments

    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return arguments

    accepted = {
        name
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in arguments.items() if k in accepted}


async def run_tool(
    func: ToolFn,
    arguments: Dict[str, Any],
    *,
    timeout: float = 30.0,
) -> Any:
    """Exécute un outil (sync ou async) avec un garde-fou de durée."""
    payload = _coerce_arguments(func, arguments)
    if inspect.iscoroutinefunction(func):
        coro = func(**payload)
    else:
        coro = asyncio.to_thread(func, **payload)
    return await asyncio.wait_for(coro, timeout=timeout)


def tool_error(name: str, message: str) -> Message:
    """Message d'erreur renvoyé au modèle, sans casser la boucle."""
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps({"error": f"{TOOL_ERROR_PREFIX}: {message}"}, ensure_ascii=False),
    }


# ------------------------------------------------------------------ harnais


class Harness:
    """Orchestre prompt système + mémoire + outils + appels LLM."""

    def __init__(self, config: LbeeConfig, memory: MemoryStore) -> None:
        self.config = config
        self.memory = memory
        self.tools: Dict[str, Tuple[ToolFn, Dict[str, Any]]] = (
            discover_tools(config.paths.tools_dir_path)
            if config.features.enable_tools
            else {}
        )
        self.schemas = openai_schemas(self.tools)
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=api_key(),
                base_url=self.config.llm.base_url,
                timeout=self.config.llm.timeout,
                max_retries=self.config.llm.max_retries,
            )
        return self._client

    # ------------------------------------------------------------ prompts

    def build_messages(self, session_id: str, message: str) -> List[Message]:
        """3 couches : système → mémoire → message courant."""
        messages: List[Message] = [{"role": "system", "content": system_prompt(self.config.paths.system_prompt_path)}]
        if self.config.memory.mode != "one-shot":
            messages.extend(self.memory.messages_for(session_id))
        messages.append({"role": "user", "content": message})
        return messages

    # -------------------------------------------------------------- appels

    async def _complete(
        self,
        messages: List[Message],
        *,
        tools: bool,
        stream: bool = False,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": messages,
        }
        kwargs.update(self.config.generation.tool_payload() if tools else self.config.generation.payload())
        if tools and self.schemas:
            kwargs["tools"] = self.schemas
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
        return await self.client.chat.completions.create(**kwargs)

    async def _compress_if_needed(self) -> None:
        """Mode shared-compressed : résume l'historique via le LLM."""
        if self.config.memory.mode != "shared-compressed":
            return
        if not self.memory.needs_compression():
            return

        history = list(self.memory._shared.messages)
        if not history:
            return

        transcript = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in history
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un compresseur de mémoire. Résume les échanges "
                    "suivants en conservant les faits, décisions, chiffres et "
                    "préférences de l'utilisateur. Réponds uniquement par le résumé."
                ),
            },
            {"role": "user", "content": transcript[:60_000]},
        ]
        try:
            response = await self._complete(messages, tools=False)
            summary = (response.choices[0].message.content or "").strip()
        except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
            log.warning("compression impossible: %s", exc)
            return
        if summary:
            self.memory.set_summary(summary)
            log.info("mémoire partagée compressée (%d messages résumés)", len(history))

    # ---------------------------------------------------------- conversation

    async def chat(self, session_id: str, message: str) -> str:
        """Un tour complet : outils éventuels, puis réponse finale."""
        clean_id = sanitize_session_id(session_id)

        async with self.memory.transaction():
            messages = self.build_messages(clean_id, message)

            if not (self.config.features.enable_tools and self.tools):
                response = await self._complete(messages, tools=False)
                answer = (response.choices[0].message.content or "").strip()
                self.memory.record_turn(clean_id, message, answer)
                await self._compress_if_needed()
                return answer

            answer = await self._agentic_loop(clean_id, messages)
            self.memory.record_turn(clean_id, message, answer)
            await self._compress_if_needed()
            return answer

    async def _agentic_loop(self, session_id: str, messages: List[Message]) -> str:
        max_iterations = self.config.features.max_tool_iterations

        for iteration in range(1, max_iterations + 1):
            response = await self._complete(messages, tools=True)
            choice = response.choices[0]
            msg = choice.message

            calls = msg.tool_calls or []
            if not calls:
                return (msg.content or "").strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )

            for call in calls:
                name = call.function.name
                log.info("itération %d/%d — appel outil: %s", iteration, max_iterations, name)

                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("les arguments doivent être un objet JSON")
                except json.JSONDecodeError as exc:
                    messages.append(tool_error(name, f"arguments JSON invalides: {exc}"))
                    continue

                entry = self.tools.get(name)
                if entry is None:
                    messages.append(
                        tool_error(name, f"outil inconnu (disponibles: {', '.join(self.tools) or 'aucun'})")
                    )
                    continue

                func, _ = entry
                try:
                    result = await run_tool(func, arguments)
                    content = json.dumps(result, ensure_ascii=False, default=str)
                except asyncio.TimeoutError:
                    messages.append(tool_error(name, "délai dépassé"))
                except ToolExecutionError as exc:
                    messages.append(tool_error(name, str(exc)))
                except Exception as exc:  # le modèle doit pouvoir se corriger
                    log.exception("outil %s en échec", name)
                    messages.append(tool_error(name, f"{type(exc).__name__}: {exc}"))

        # Nombre maximal d'allers-retours atteint : on termine proprement.
        try:
            response = await self._complete(messages, tools=False)
            return (response.choices[0].message.content or "").strip()
        except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(f"limite d'itérations d'outils atteinte ({max_iterations}): {exc}") from exc

    # ------------------------------------------------------------ streaming

    async def stream(self, session_id: str, message: str) -> AsyncIterator[str]:
        """Flux SSE : les outils sont résolus avant d'émettre le premier token.

        Aucun fragment de ``tool_calls`` n'est diffusé : on bascule en mode
        non-streaming le temps de la décision, puis on stream la réponse.
        """
        clean_id = sanitize_session_id(session_id)

        async with self.memory.transaction():
            messages = self.build_messages(clean_id, message)
            answer: Optional[str] = None

            if self.config.features.enable_tools and self.tools:
                try:
                    answer = await self._agentic_loop(clean_id, messages)
                except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
                    raise RuntimeError(f"erreur LLM: {exc}") from exc

            if answer is not None:
                # Réponse déjà connue : émise en un seul événement.
                yield sse_event("delta", answer)
            else:
                try:
                    stream = await self._complete(messages, tools=False, stream=True)
                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, "content", None)
                        if piece:
                            yield sse_event("delta", piece)
                except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
                    yield sse_event("error", str(exc))
                    return

            self.memory.record_turn(clean_id, message, answer or "")
            try:
                await self._compress_if_needed()
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
                log.warning("compression différée: %s", exc)

            yield sse_event("done", "[DONE]")


def sse_event(event: str, data: str) -> str:
    payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


__all__ = [
    "Harness",
    "sse_event",
    "ToolExecutionError",
    "ToolNotFound",
    "discover_tools",
    "openai_schemas",
    "run_tool",
]