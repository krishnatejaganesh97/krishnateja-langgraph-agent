"""FastAPI backend for the Teja LangGraph Agent.

Endpoints (same pattern as Product 1, so Teja can compare the two):
  GET  /health   — is the service up, which mode (mock/openai/azure)?
  POST /run      — {task} -> {final (structured), plan, log, tool_calls, revisions}
  GET  /tasks    — sample tasks to try
  GET  /         — serves the frontend UI (frontend/index.html)

Run from the project root:
  python -m uvicorn backend.main:app --reload --port 8001
  then open http://localhost:8001
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from backend.agent import AgentOutput, detect_mode, run_agent  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Teja LangGraph Agent",
    description="LangGraph agentic workflow: plan -> research -> draft -> critique -> final, with tool calls and structured output. Mock mode runs without an API key.",
    version="1.0.0",
)

SAMPLE_TASKS = [
    "Summarize Teja's AI experience for an AI Engineer role",
    "Compare RAG vs fine-tuning for a doc chatbot",
    "Where does Teja study for his PhD?",
    "What did Teja build at Elite Tech Info?",
    "What programming languages does Teja use?",
]


class RunRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=1000,
                      description="The research / job-search task for the agent.")


class RunResponse(BaseModel):
    task: str
    mode: str
    plan: list[str]
    tool_calls: list[dict]
    evidence: list[dict]
    critique: str
    revisions: int
    log: list[dict]
    final: AgentOutput


@app.get("/health")
def health():
    return {"status": "ok", "mode": detect_mode(),
            "graph": "plan -> research -> draft -> critique -> (revise | finalize)"}


@app.get("/tasks")
def tasks():
    return {"tasks": SAMPLE_TASKS}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest):
    return RunResponse(**run_agent(req.task))


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
