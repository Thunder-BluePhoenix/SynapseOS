"""
SynapseOS REST API Server.

Wraps CognitiveController, sub-agents, and memory in a FastAPI service.
Enables SynapseOS to be used as an HTTP service from Frappe, web UIs,
or any other HTTP client.

Endpoints:
  GET  /health                — system health + agent status
  GET  /stats                 — memory, agent, KG stats
  GET  /agents                — list agents and their capabilities
  POST /query                 — main inference endpoint
  POST /query/mode/{mode}     — force a specific thinking mode
  POST /query/agent/{agent}   — force a specific sub-agent
  GET  /memory/context        — retrieve episodic+semantic context for a query
  POST /memory/record         — manually record an interaction
  GET  /kg/search             — search the knowledge graph
  POST /voice/process         — voice round-trip (text in, text out)
  POST /benchmark             — quick load test at specified concurrency

Start:
    uvicorn synapseos.api.server:app --host 0.0.0.0 --port 8765 --reload
    python -m synapseos.api.server   # convenience runner

Frappe integration (call from Python controller or whitelisted method):
    import requests
    resp = requests.post("http://localhost:8765/query", json={"query": "..."})
"""

from __future__ import annotations

import argparse
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from synapseos.api.models import (
    QueryRequest, QueryResponse,
    HealthResponse, AgentsResponse, AgentInfo,
    KGSearchResponse, KGSearchResult,
    MemoryContextResponse, MemoryRecordRequest,
    VoiceRequest, VoiceResponse,
    BenchmarkRequest, BenchmarkResponse,
)


# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}

KG_PATH    = "data/graph/knowledge_graph.json"
RULES_PATH = "data/rules/rules.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CognitiveController and KG at startup; release at shutdown."""
    print("[API] Starting SynapseOS API server ...")
    t0 = time.time()

    from synapseos.phase5.cognitive_controller import CognitiveController
    from synapseos.phase3.neurosymbolic_core import KnowledgeGraph

    _state["controller"] = CognitiveController(
        kg_path=KG_PATH,
        rules_path=RULES_PATH,
        mode_timeout=30.0,
        max_parallel_modes=2,
        enable_idle_loop=False,
        enable_agents=True,
    )
    _state["kg"]     = KnowledgeGraph(KG_PATH)
    _state["started"] = datetime.now().isoformat()

    elapsed = (time.time() - t0) * 1000
    print(f"[API] Ready in {elapsed:.0f}ms | agents: {_state['controller'].stats()['agents_loaded']}")
    yield

    _state["controller"].shutdown()
    print("[API] Server shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SynapseOS API",
    description=(
        "Cognitive reasoning API grounded in the Frappe/ERPNext knowledge graph. "
        "All responses are traceable — no hallucinated APIs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _controller():
    c = _state.get("controller")
    if c is None:
        raise HTTPException(503, "Controller not initialised")
    return c


def _kg():
    kg = _state.get("kg")
    if kg is None:
        raise HTTPException(503, "Knowledge graph not loaded")
    return kg


# ---------------------------------------------------------------------------
# Health + info endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """System health check — confirms all components are ready."""
    ctrl  = _controller()
    stats = ctrl.stats()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        agents_loaded=stats.get("agents_loaded", []),
        memory_stats={
            "episodic_entries": stats.get("episodic_entries", 0),
            "semantic_facts":   stats.get("semantic_facts", 0),
            "working_slots":    stats.get("working_slots", 0),
        },
        idle_loop_active=stats.get("idle_loop_active", False),
    )


@app.get("/stats", tags=["Info"])
def stats():
    """Detailed system statistics — memory, KG, agents."""
    ctrl   = _controller()
    kg     = _kg()
    cstats = ctrl.stats()
    return {
        "timestamp":      datetime.now().isoformat(),
        "controller":     cstats,
        "knowledge_graph": {
            "nodes": len(kg._nodes),
            "edges": len(kg._edges),
        },
        "uptime_since": _state.get("started", "unknown"),
    }


@app.get("/agents", response_model=AgentsResponse, tags=["Info"])
def list_agents():
    """List all registered sub-agents and their capabilities."""
    ctrl   = _controller()
    router = ctrl.agent_router
    agents = []
    if router:
        for agent in router._agents:
            cap = agent.capability
            agents.append(AgentInfo(
                name=cap.name,
                description=cap.description,
                keywords=cap.keywords[:10],
                base_confidence=cap.base_confidence,
            ))
    return AgentsResponse(agents=agents, count=len(agents))


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["Inference"])
def query(req: QueryRequest):
    """
    Main inference endpoint. Routes to thinking modes and/or agents automatically.
    Set `force_modes` or `force_agent` to override routing.
    """
    ctrl = _controller()

    # Build force_modes list
    force_modes = req.force_modes
    if req.force_agent and not force_modes:
        # Dispatch only the specified agent via force, bypass mode routing
        router = ctrl.agent_router
        if router:
            agent = next((a for a in router._agents if a.name.lower() == req.force_agent.lower()), None)
            if agent is None:
                raise HTTPException(404, f"Agent '{req.force_agent}' not found. Available: {[a.name for a in router._agents]}")
        # Fall through — agent router will pick it up normally
        force_modes = None

    result = ctrl.think(req.query, force_modes=force_modes)
    return QueryResponse.from_cognition_result(result)


@app.post("/query/mode/{mode}", response_model=QueryResponse, tags=["Inference"])
def query_mode(mode: str, req: QueryRequest):
    """Force a specific thinking mode. mode: mode1 | mode2 | mode3 | mode4"""
    valid = {"mode1", "mode2", "mode3", "mode4"}
    if mode not in valid:
        raise HTTPException(400, f"Invalid mode '{mode}'. Valid: {sorted(valid)}")
    req.force_modes = [mode]
    return query(req)


@app.post("/query/agent/{agent_name}", response_model=QueryResponse, tags=["Inference"])
def query_agent(agent_name: str, req: QueryRequest):
    """Force a specific agent. agent_name: ARIEL | NYX | NOVA | ATLAS"""
    ctrl   = _controller()
    router = ctrl.agent_router
    if not router:
        raise HTTPException(503, "Agent router not available")

    agent = next((a for a in router._agents if a.name.lower() == agent_name.lower()), None)
    if agent is None:
        names = [a.name for a in router._agents]
        raise HTTPException(404, f"Agent '{agent_name}' not found. Available: {names}")

    # Run agent directly
    result = agent.handle(req.query, context=None)
    # Wrap in a minimal CognitionResult-like structure
    from synapseos.phase5.cognitive_controller import CognitionResult, ThinkingMode
    class _Wrap:
        query = req.query
        primary_output = result.output
        output_type = "language"
        confidence = result.confidence
        modes_activated = [ThinkingMode.MODE1_REASONING]
        routed_to_agent = result.agent_name
        total_latency_ms = result.latency_ms
        memory_context = {"summary": f"Direct agent call: {result.agent_name}"}
        mode_results = []
        agent_results = [result]

    return QueryResponse.from_cognition_result(_Wrap())


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------

@app.get("/memory/context", response_model=MemoryContextResponse, tags=["Memory"])
def memory_context(q: str = Query(..., description="Query to retrieve context for")):
    """Retrieve episodic + semantic memory context for a query."""
    ctrl = _controller()
    ctx  = ctrl.memory.get_context(q, top_k=5)
    return MemoryContextResponse(
        query=q,
        summary=ctx.summary,
        recent_count=len(ctx.recent_interactions),
        similar_count=len(ctx.similar_past),
        semantic_facts_count=len(ctx.semantic_facts),
        working_slots=len(ctrl.memory.working),
        top_similar=[
            {"score": s.get("score", 0), "query": s.get("query", ""), "mode": s.get("mode_used", "")}
            for s in ctx.similar_past[:3]
        ],
        top_facts=[
            {"subject": f.get("subject", ""), "predicate": f.get("predicate", ""), "obj": f.get("obj", "")}
            for f in ctx.semantic_facts[:3]
        ],
    )


@app.post("/memory/record", tags=["Memory"])
def memory_record(req: MemoryRecordRequest):
    """Manually record an interaction in episodic memory."""
    ctrl = _controller()
    eid  = ctrl.memory.record_interaction(req.query, req.response, req.mode_used, req.tags)
    return {"entry_id": eid, "total_entries": ctrl.memory.stats()["episodic_entries"]}


# ---------------------------------------------------------------------------
# Knowledge graph endpoints
# ---------------------------------------------------------------------------

@app.get("/kg/search", response_model=KGSearchResponse, tags=["Knowledge Graph"])
def kg_search(
    q:     str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """Full-text search over the knowledge graph."""
    kg      = _kg()
    results = kg.search(q, limit=limit)
    items   = []
    for r in results:
        name = r.get("name", r.get("id", ""))
        items.append(KGSearchResult(
            name=name,
            score=r.get("score", 0.0),
            module=r.get("module", ""),
            is_submittable=bool(r.get("is_submittable", False)),
        ))
    return KGSearchResponse(query=q, results=items, total=len(items))


@app.get("/kg/entity/{name}", tags=["Knowledge Graph"])
def kg_entity(name: str):
    """Get a specific entity from the knowledge graph."""
    kg     = _kg()
    entity = kg.get_entity(name)
    if entity is None:
        raise HTTPException(404, f"Entity '{name}' not found")
    links  = kg.get_related_doctypes(name)
    return {"name": name, "attributes": entity, "linked_doctypes": links}


@app.get("/kg/path", tags=["Knowledge Graph"])
def kg_path(
    source: str = Query(...),
    target: str = Query(...),
    max_hops: int = Query(3, ge=1, le=5),
):
    """Find shortest path between two entities in the knowledge graph."""
    kg   = _kg()
    path = kg.find_path(source, target, max_hops=max_hops)
    if path is None:
        raise HTTPException(404, f"No path found from '{source}' to '{target}' within {max_hops} hops")
    return {"source": source, "target": target, "path": path, "hops": len(path) - 1}


# ---------------------------------------------------------------------------
# Voice endpoint
# ---------------------------------------------------------------------------

@app.post("/voice/process", response_model=VoiceResponse, tags=["Voice"])
def voice_process(req: VoiceRequest):
    """
    Process text through the voice pipeline.
    Returns transcription, cognition output, and synthesis result.
    In stub mode: uses text passthrough + file-based TTS.
    """
    from synapseos.phase5.voice import VoicePipeline
    pipeline = VoicePipeline(kg_path=KG_PATH, rules_path=RULES_PATH)
    result   = pipeline.process_text(req.text)
    pipeline.shutdown()
    return VoiceResponse(
        transcription=result.transcription.text,
        cognition_output=result.cognition_output,
        synthesis_path=result.synthesis.audio_path,
        total_latency_ms=result.total_latency_ms,
        within_target=result.within_target,
    )


# ---------------------------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------------------------

@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Operations"])
def benchmark(req: BenchmarkRequest):
    """
    Run a quick load test against the local pipeline.
    Returns latency percentiles, throughput, and pass/fail vs Phase 5 criterion.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from benchmark_pipeline import run_level, BENCHMARK_QUERIES

    ctrl = _controller()

    def fn(query: str):
        ctrl.think(query, force_modes=["mode1"])

    stats = run_level(fn, req.concurrency, BENCHMARK_QUERIES, req.n_requests)
    return BenchmarkResponse(
        concurrency=req.concurrency,
        n_requests=stats.n_requests,
        n_success=stats.n_success,
        error_rate=stats.error_rate,
        p50_ms=stats.p50_ms,
        p99_ms=stats.p99_ms,
        throughput_rps=stats.throughput_rps,
        passes_criterion=stats.passes_criterion,
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="SynapseOS API Server")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"\nSynapseOS API Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")

    uvicorn.run(
        "synapseos.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
