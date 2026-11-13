"""Outil de lecture de la base de connaissances (/kb)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

DESCRIPTION = "Lit un fichier de la base de connaissances (kb/)."

SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_kb",
        "description": (
            "Lit le contenu d'un fichier présent dans la base de connaissances (dossier kb/). "
            "Utilise cet outil pour consulter la charte de qualité ou tout autre document de référence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Nom du fichier à lire (ex: 'charte-qualite.md'). Ne pas mettre de chemin.",
                }
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
}

# Dossier kb monté dans le conteneur
KB_DIR = Path("/app/kb")


def execute(filename: str) -> Dict[str, Any]:
    if not filename or not isinstance(filename, str):
        raise ValueError("filename est requis")

    # Sécurité : on interdit les chemins relatifs (../ etc.)
    clean_name = Path(filename).name
    if clean_name != filename:
        raise ValueError("nom de fichier invalide (pas de chemin autorisé)")

    path = KB_DIR / clean_name

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable dans la KB : {clean_name}")

    if not path.is_file():
        raise ValueError(f"{clean_name} n'est pas un fichier")

    # Limite de taille pour éviter de saturer le contexte (≈ 100 ko)
    if path.stat().st_size > 100_000:
        raise ValueError(f"Fichier trop volumineux (> 100 Ko) : {clean_name}")

    content = path.read_text(encoding="utf-8")

    return {
        "filename": clean_name,
        "content": content,
    }
