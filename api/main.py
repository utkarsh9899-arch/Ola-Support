"""FastAPI deployment (Task 11) with JSON-Lines request logging (Task 12).

POST /ask           one question -> guarded, crew-answered, schema-validated SupportResponse
POST /add-document  add a policy document to both ChromaDB collections
GET  /health        liveness
WS   /ws/chat       real-time multi-turn chat with session memory; survives client disconnects

Run:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (offline / no-telemetry env first)

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from ola_agent import guardrails  # noqa: E402
from ola_agent.budget import BudgetExceededError  # noqa: E402
from ola_agent.logging_utils import log_request, new_trace_id  # noqa: E402
from ola_agent.memory import ask  # noqa: E402
from ola_agent.pipeline import PipelineResult  # noqa: E402
from ola_agent.rag.index import add_document, build_index  # noqa: E402
from ola_agent.rag.kb import Document, split_sentences  # noqa: E402
from ola_agent.schemas import ReviewVerdict, SupportResponse  # noqa: E402


# --- Pydantic request/response models --------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="The customer's message.")
    session_id: str | None = Field(default=None, max_length=64, description="Reuse to keep conversation memory.")


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    response: SupportResponse
    review: ReviewVerdict | None = Field(default=None, description="Autogen review-stage verdict, when a draft was reviewed.")
    tokens_used: int = Field(description="Metered tokens across every model call (budget cap applies).")
    latency_ms: float


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9_]{3,64}$", description="e.g. 13_lost_item_policy")
    title: str = Field(min_length=3, max_length=120)
    text: str = Field(min_length=20, description="2-5 sentences of policy text.")


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    sentences: int
    chunks_upserted: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    mock_llm: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(build_index)  # idempotent upsert of the KB into both collections
    yield


app = FastAPI(title="Ola Support Agent", version="1.0", lifespan=lifespan)


# --- one JSON-Lines entry per HTTP request ---------------------------------------
@app.middleware("http")
async def request_logging(request: Request, call_next):
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    request.state.log_fields = {}
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Trace-ID"] = trace_id
        return response
    finally:
        log_request(
            "http_request", trace_id, (time.perf_counter() - start) * 1000,
            **{"http.request.method": request.method, "url.path": request.url.path,
               "http.response.status_code": status, **request.state.log_fields},
        )


def _result_fields(result: PipelineResult, session_id: str) -> dict:
    r = result.response
    return {"session.id": session_id, "query.masked": result.masked_message, "response.intent": r.intent,
            "response.refused": r.refused, "guardrail.flags": r.guardrail_flags}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", mock_llm=config.MOCK_LLM)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    start = time.perf_counter()
    session_id = body.session_id or f"http-{uuid.uuid4().hex[:12]}"
    try:
        result = ask(body.query, session_id)
    except BudgetExceededError as exc:
        request.state.log_fields = {"session.id": session_id, "query.masked": body.query[:200],
                                    "budget.rejected": True, "budget.tokens": exc.tokens, "budget.limit": exc.limit}
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    request.state.log_fields = {**_result_fields(result, session_id), "budget.tokens_used": result.tokens_used}
    return AskResponse(trace_id=request.state.trace_id, session_id=session_id, response=result.response,
                       review=result.review, tokens_used=result.tokens_used,
                       latency_ms=round((time.perf_counter() - start) * 1000, 2))


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(body: AddDocumentRequest, request: Request) -> AddDocumentResponse:
    text, _ = guardrails.mask_pii(body.text)  # no raw phone numbers in the knowledge base either
    counts = add_document(Document(doc_id=body.doc_id, title=body.title, text=text))
    request.state.log_fields = {"document.id": body.doc_id}
    return AddDocumentResponse(trace_id=request.state.trace_id, doc_id=body.doc_id,
                               sentences=len(split_sentences(text)), chunks_upserted=counts)


# --- WebSocket: real-time multi-turn chat ----------------------------------------
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, session_id: str | None = None):
    await websocket.accept()
    session_id = session_id or f"ws-{uuid.uuid4().hex[:12]}"
    turns = 0
    try:
        while True:
            message = await websocket.receive_text()
            trace_id, start = new_trace_id(), time.perf_counter()
            try:
                result = await run_in_threadpool(ask, message, session_id)
            except BudgetExceededError as exc:
                log_request("ws_message", trace_id, (time.perf_counter() - start) * 1000,
                            **{"url.path": "/ws/chat", "session.id": session_id, "budget.rejected": True,
                               "budget.tokens": exc.tokens, "budget.limit": exc.limit})
                await websocket.send_json({"trace_id": trace_id, "session_id": session_id, "error": str(exc)})
                continue
            turns += 1
            log_request("ws_message", trace_id, (time.perf_counter() - start) * 1000,
                        **{"url.path": "/ws/chat", **_result_fields(result, session_id), "ws.turn": turns})
            await websocket.send_json({"trace_id": trace_id, "session_id": session_id, "turn": turns,
                                       "response": result.response.model_dump()})
    except (WebSocketDisconnect, RuntimeError, OSError) as exc:
        # Client went away (possibly mid-answer). Log it, drop this connection only; the server keeps serving.
        log_request("ws_disconnect", new_trace_id(), 0.0,
                    **{"url.path": "/ws/chat", "session.id": session_id, "ws.turns_completed": turns,
                       "ws.disconnect": type(exc).__name__})
