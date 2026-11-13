"""Chargement et validation de la configuration (Pydantic)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MEMORY_MODES = ("one-shot", "personal", "shared-compressed")


class ConfigError(RuntimeError):
    """Erreur de configuration détectée au démarrage."""


class StrictModel(BaseModel):
    """Modèle Pydantic strict : les clés inconnues sont refusées."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LlmConfig(StrictModel):
    """Client LLM compatible OpenAI."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = Field(default=60, ge=1, le=3600)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def _clean_base_url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url doit commencer par http:// ou https://")
        return value.rstrip("/")

    @field_validator("model")
    @classmethod
    def _clean_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("llm.model ne peut pas être vide")
        return value


class GenerationConfig(StrictModel):
    """Paramètres de génération envoyés à l'API."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=200_000)

    # Uniquement des paramètres non standards, ex. {"top_k": 40}.
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    extra_body_tools: Dict[str, Any] = Field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        """Corps de requête « compatible OpenAI » (sans les messages)."""
        data: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        # Les réglages propriétaires (top_k, repetition_penalty, ...) restent
        # isolés dans extra_body : l'API OpenAI ne les accepte pas au premier
        # niveau et répondrait 400.
        data.update(self.extra_body)
        return data

    def tool_payload(self) -> Dict[str, Any]:
        """Corps de requête utilisé pour l'étape de décision d'outils."""
        data = self.payload()
        data.update(self.extra_body_tools)
        return data


class MemoryConfig(StrictModel):
    """Politique de mémoire de session."""

    mode: Literal["one-shot", "personal", "shared-compressed"] = "one-shot"
    max_history_messages: int = Field(default=20, ge=0, le=2000)
    compression_threshold: int = Field(default=6000, ge=500, le=400_000)
    keep_last_messages_after_compression: int = Field(default=6, ge=0, le=200)

    # Renseigné au démarrage depuis paths.kb_dir.
    kb_dir: Optional[Path] = None

    @property
    def persist(self) -> bool:
        return self.mode in ("personal", "shared-compressed")

    def prepare(self) -> None:
        if not self.persist or self.kb_dir is None:
            return
        try:
            self.kb_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"impossible de créer {self.kb_dir}: {exc}") from exc


class FeaturesConfig(StrictModel):
    """Interrupteurs fonctionnels."""

    enable_tools: bool = True
    max_tool_iterations: int = Field(default=5, ge=1, le=25)
    enable_streaming: bool = False


class Paths:
    """Résolution des chemins (absolus ou relatifs à /app)."""

    APP_ROOT = Path(os.environ.get("LBE_ROOT", "/app"))

    def __init__(
        self,
        system_prompt: str = "system_prompt.md",
        tools_dir: str = "class-tools",
        kb_dir: str = "kb",
    ) -> None:
        self.system_prompt = system_prompt
        self.tools_dir = tools_dir
        self.kb_dir = kb_dir
        self.system_prompt_path: Path = Path(system_prompt)
        self.tools_dir_path: Path = Path(tools_dir)
        self.kb_dir_path: Path = Path(kb_dir)

    def resolve(self) -> None:
        self.system_prompt_path = self._absolutize(self.system_prompt)
        self.tools_dir_path = self._absolutize(self.tools_dir)
        self.kb_dir_path = self._absolutize(self.kb_dir)

        if not self.system_prompt_path.is_file():
            raise ConfigError(f"prompt système introuvable: {self.system_prompt_path}")

    def _absolutize(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.APP_ROOT / path


class LbeeConfig(StrictModel):
    """Racine de la configuration."""

    llm: LlmConfig = Field(default_factory=LlmConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    paths: Paths = Field(default_factory=Paths)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _link_memory_paths(self) -> "LbeeConfig":
        self.paths.resolve()
        self.memory.kb_dir = self.paths.kb_dir_path
        return self

    @classmethod
    def from_file(cls, path: Path) -> "LbeeConfig":
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"config.yml introuvable: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"config.yml invalide: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError("config.yml doit être un mapping à la racine")

        try:
            return cls(**raw)
        except Exception as exc:  # pydantic.ValidationError et dérivés
            raise ConfigError(f"configuration invalide: {exc}") from exc


def api_key() -> str:
    """Clé API lue exclusivement depuis l'environnement."""
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        raise ConfigError("API_KEY manquante (voir .env.example)")
    return key


def system_prompt(path: Path) -> str:
    """Contenu du prompt système."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"lecture impossible: {path}: {exc}") from exc
    if not text:
        raise ConfigError(f"prompt système vide: {path}")
    return text


def describe_tools(tools: Dict[str, Any]) -> List[str]:
    return sorted(tools)


__all__ = [
    "ConfigError",
    "FeaturesConfig",
    "GenerationConfig",
    "LbeeConfig",
    "LlmConfig",
    "MEMORY_MODES",
    "MemoryConfig",
    "Paths",
    "api_key",
    "system_prompt",
]