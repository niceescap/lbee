"""Config, invariants et singletons partagés."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import CONFIG, ConfigError, Paths


def die(message: str, code: int = 1) -> None:
    """Affiche une erreur fatale et arrête le processus."""
    print(f"[lbee] FATAL: {message}", file=sys.stderr)
    sys.exit(code)


def init() -> None:
    """Etablit les invariants du harnais.

    - charge et valide ``config.yml``
    - vérifie la présence de ``API_KEY``
    - résout le prompt système
    - prépare les répertoires de travail
    """
    try:
        CONFIG.load()
    except ConfigError as exc:  # pragma: no cover - chemin d'arrêt
        die(str(exc))

    if not os.environ.get("API_KEY", "").strip():
        die("API_KEY manquante (voir .env.example).")

    try:
        CONFIG.paths.resolve()
    except ConfigError as exc:  # pragma: no cover - chemin d'arrêt
        die(str(exc))

    CONFIG.memory.prepare()
    print(
        f"[lbee] prête — modèle={CONFIG.llm.model} mémoire={CONFIG.memory.mode} "
        f"outils={CONFIG.features.enable_tools} streaming={CONFIG.features.enable_streaming}"
    )


__all__ = ["CONFIG", "Paths", "ConfigError", "die", "init"]