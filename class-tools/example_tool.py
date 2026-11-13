"""Outil d'exemple : calcul arithmétique sûr.

Contrat d'un outil lbee :
    * ``SCHEMA``       → déclaration OpenAI function-calling (obligatoire)
    * ``execute(**kw)`` → implémentation synchrone ou asynchrone (obligatoire)
    * ``DESCRIPTION``  → documentation optionnelle

Toute exception levée est renvoyée au modèle comme message d'outil
d'erreur : le harnais n'interrompt jamais la conversation.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict

DESCRIPTION = "Évalue une expression arithmétique simple (+ - * / % ** et parenthèses)."

SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Calcule une expression arithmétique élémentaire et renvoie le "
            "résultat numérique. N'accepte que des nombres, les opérateurs "
            "+ - * / // % ** et des parenthèses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression à évaluer, ex. '12 * (3 + 4)'.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("expression non autorisée")


def execute(expression: str) -> Dict[str, Any]:
    """Évalue ``expression`` et renvoie ``{"result": ...}``."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("argument 'expression' manquant ou vide")
    if len(expression) > 200:
        raise ValueError("expression trop longue (200 caractères max)")
    tree = ast.parse(expression, mode="eval")
    return {"result": _eval(tree)}