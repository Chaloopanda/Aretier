"""
app/server.py
-------------
FastAPI backend server for Arétier ASBOS.
Serves the HTML frontend and exposes the drop pipeline as a REST API.
"""

import os
import sys
import uuid
import json
import asyncio
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status": "running"|"done"|"error", "result": {...}, "error": "..." } }
_jobs: dict = {}

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = Path("./drop_history.json")


def _load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_to_history(result: dict):
    history = _load_history()
    entry = {
        "drop_id": result.get("drop_id"),
        "season": result.get("season"),
        "timestamp": datetime.utcnow().isoformat(),
        "aesthetic_keywords": result.get("aesthetic_keywords", []),
        "suggested_price": result.get("suggested_price"),
        "approval_score": result.get("approval_score"),
        "recommended_drop_time": result.get("recommended_drop_time"),
        "product_description": result.get("product_description"),
        "design_brief": result.get("design_brief"),
        "iteration_count": result.get("iteration_count", 0),
        "has_image": result.get("has_image", False),
        "image_filename": result.get("image_filename"),
    }
    history.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(history[:50], indent=2), encoding="utf-8")


def _run_pipeline_sync(job_id: str, drop_id: str, season: str):
    """Run the drop pipeline synchronously in a thread."""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["completed_steps"] = []
        sys.path.insert(0, str(Path(__file__).parent.parent))

        def step_callback(node_name: str):
            _jobs[job_id]["completed_steps"].append(node_name)

        from graph.graph_builder import run_drop_pipeline
        result = run_drop_pipeline(
            drop_id=drop_id,
            season=season,
            thread_id=str(uuid.uuid4()),
            on_step=step_callback,
        )

        # Serialise image path for JSON
        if result.get("design_image_b64"):
            result["has_image"] = True
        else:
            result["has_image"] = False
        result.pop("design_image_b64", None)  # Don't send raw b64 over API
        result.pop("messages", None)

        _save_to_history(result)
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Arétier ASBOS", version="1.0.0")

# Serve artifacts (images) as static files
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


class TriggerRequest(BaseModel):
    season: str = "Summer"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1>")


# Global executor to enforce true single-concurrency
import concurrent.futures
_pipeline_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

@app.post("/api/drop/trigger")
async def trigger_drop(req: TriggerRequest, background_tasks: BackgroundTasks):
    drop_id = str(uuid.uuid4())[:8]
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None, "error": None, "drop_id": drop_id, "completed_steps": []}

    # Run pipeline in background thread (true singleton executor)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_pipeline_executor, _run_pipeline_sync, job_id, drop_id, req.season)

    return {"job_id": job_id, "drop_id": drop_id, "status": "running"}


@app.get("/api/drop/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "completed_steps": job.get("completed_steps", []),
        "result": job.get("result") if job["status"] == "done" else None,
    }


@app.get("/api/drops/history")
async def get_history():
    return {"history": _load_history()}


@app.get("/api/image/{drop_id}/{filename}")
async def get_image(drop_id: str, filename: str):
    image_path = ARTIFACTS_DIR / drop_id / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(image_path), media_type="image/png")


@app.get("/api/health")
async def health():
    from tools.llm_provider import get_provider_name
    return {"status": "ok", "provider": get_provider_name()}
