"""Configuration chargée, invariants et singletons partagés."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import ConfigError, LbeeConfig, Paths, api_key

CONFIG_PATH = Path(os.environ.get("LBE_CONFIG", str(Paths.APP_ROOT / "config.yml")))


def die(message: str, code: int = 1) -> None:
    """Affiche une erreur fatale et arrête le processus."""
    print(f"[lbee] FATAL: {message}", file=sys.stderr)
    sys.exit(code)


def load_config(path: Path | None = None) -> LbeeConfig:
    """Charge et valide ``config.yml`` (Pydantic strict)."""
    return LbeeConfig.from_file(path or CONFIG_PATH)


def init() -> LbeeConfig:
    """Établit les invariants du harnais.

    - charge et valide ``config.yml``
    - vérifie la présence de ``API_KEY``
    - résout le prompt système et les répertoires
    - prépare le répertoire de persistance ``kb``
    """
    global CONFIG

    try:
        config = load_config()
    except ConfigError as exc:  # pragma: no cover - chemin d'arrêt
        die(str(exc))

    try:
        api_key()
    except ConfigError as exc:
        die(str(exc))

    try:
        config.paths.resolve()
        config.memory.prepare()
    except ConfigError as exc:  # pragma: no cover - chemin d'arrêt
        die(str(exc))

    CONFIG = config
    print(
        f"[lbee] prête — modèle={config.llm.model} mémoire={config.memory.mode} "
        f"outils={config.features.enable_tools} streaming={config.features.enable_streaming}",
        flush=True,
    )
    return config


# Renseigné par init() ; reste None tant que le service n'a pas démarré.
CONFIG: LbeeConfig | None = None


__all__ = ["CONFIG", "CONFIG_PATH", "ConfigError", "die", "init", "load_config"]