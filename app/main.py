"""API HTTP minimale : POST /chat et GET /health."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from . import state
from .config import ConfigError
from .harness import Harness, sse_event
from .memory import MemoryStore, sanitize_session_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lbee.api")

state.init()

app = FastAPI(title="lbee", version="0.1.0", description="Harnais LLM minimaliste")

memory = MemoryStore(state.CONFIG.memory, state.CONFIG.paths.kb_dir_path)
harness = Harness(state.CONFIG, memory)


class ChatRequest(BaseModel):
    """Corps de ``POST /chat``."""

    message: str = Field(..., min_length=1, description="Message de l'utilisateur")
    session_id: str = Field(default="default", description="Identifiant de session")
    stream: Optional[bool] = Field(
        default=None,
        description="Force (ou non) le streaming, sinon valeur de config.yml",
    )


class ChatResponse(BaseModel):
    """Réponse standard en mode non-streaming."""

    response: str
    session_id: str
    model: str
    memory_mode: str


@app.get("/health", summary="Disponibilité du service")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": state.CONFIG.llm.model,
            "memory": state.CONFIG.memory.mode,
            "tools": len(harness.tools),
            "streaming": state.CONFIG.features.enable_streaming,
        }
    )


@app.post("/chat", summary="Conversation agentique")
async def chat(payload: ChatRequest) -> object:
    session_id = sanitize_session_id(payload.session_id)
    use_stream = (
        state.CONFIG.features.enable_streaming if payload.stream is None else payload.stream
    )

    if not use_stream:
        try:
            answer = await harness.chat(session_id, payload.message)
        except RateLimitError as exc:
            raise HTTPException(status_code=429, detail=f"limite de débit LLM: {exc}") from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise HTTPException(status_code=504, detail=f"LLM injoignable: {exc}") from exc
        except (APIError, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail=f"erreur LLM: {exc}") from exc

        return ChatResponse(
            response=answer,
            session_id=session_id,
            model=state.CONFIG.llm.model,
            memory_mode=state.CONFIG.memory.mode,
        )

    async def event_stream():
        try:
            async for event in harness.stream(session_id, payload.message):
                yield event
        except ConfigError as exc:
            log.error("configuration: %s", exc)
        except Exception as exc:  # garder le flux SSE valide jusqu'au bout
            log.exception("streaming interrompu")
            yield sse_event("error", str(exc))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/memory/reset", summary="Réinitialise la mémoire")
async def reset_memory(session_id: str = "default") -> JSONResponse:
    memory.clear(sanitize_session_id(session_id))
    return JSONResponse({"status": "ok", "session_id": sanitize_session_id(session_id)})