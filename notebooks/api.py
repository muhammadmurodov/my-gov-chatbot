"""
OpenAI-compatible FastAPI wrapper around the gated RAG pipeline.

Exposes the two endpoints an OpenAI-compatible chat UI needs:
  GET  /v1/models              -> so Open WebUI can discover the "mygov-rag" model
  POST /v1/chat/completions    -> the gated answer path (streaming or not)

The last user message is treated as the question and routed through
rerank.answer -> classifier gate + grounding floor + rerank + LLM. Off-topic /
ungrounded queries come back as the fixed refusal, same as before.

Run from the repo root (so `data/` paths and the bare `import rerank` chain resolve):
    uvicorn api:app --app-dir notebooks --port 8000

Then point Open WebUI at http://<host>:8000/v1 (see docker-compose.yml), or curl it:
    curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"mygov-rag","messages":[{"role":"user","content":"..."}]}'
"""
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import rerank  # gated answer path: classifier -> grounding floor -> rerank -> LLM

app = FastAPI(title="my.gov.uz RAG", version="1.0")

MODEL_ID = "mygov-rag"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    # accepted for OpenAI-client compatibility; the pipeline sets its own values
    temperature: float | None = None
    stream: bool | None = False


def _split_question_and_history(messages: list[Message]):
    """Find the last non-empty user message (the question) and return it together
    with the prior turns as [{"role","content"}] history. Open WebUI already sends
    the full conversation each request, so we just use it instead of discarding it.
    Returns (None, []) if there is no usable user message."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "user" and m.content.strip():
            history = [{"role": p.role, "content": p.content} for p in messages[:i]]
            return m.content.strip(), history
    return None, []


def _sources(rows):
    """Which services grounded the answer (empty on refusal)."""
    return [{"service_id": r[0], "title": r[1], "url": r[2],
             "score": round(float(r[5]), 4)} for r in rows]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    """Open WebUI calls this to populate its model picker."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "my.gov.uz",
        }],
    }


def _stream_response(model, answer_text, rows):
    """Emit the answer as an OpenAI-style SSE stream. The pipeline produces the
    whole answer at once, so we send it as a single content delta then close;
    this satisfies streaming clients (Open WebUI defaults to stream=true)."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def base(delta, finish_reason=None):
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def gen():
        yield f"data: {json.dumps(base({'role': 'assistant'}))}\n\n"
        yield f"data: {json.dumps(base({'content': answer_text}))}\n\n"
        # non-standard extra: which services grounded the answer (empty on refusal)
        done = base({}, finish_reason="stop")
        done["sources"] = _sources(rows)
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    question, history = _split_question_and_history(req.messages)
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no user message with content",
                               "type": "invalid_request_error"}},
        )

    answer_text, rows = rerank.answer(question, history=history)

    if req.stream:
        return _stream_response(req.model, answer_text, rows)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": "stop",
        }],
        # non-standard extra: which services grounded the answer (empty on refusal)
        "sources": _sources(rows),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
