"""Pydantic request/response models for the SynapseOS REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    force_modes: list[str] | None = Field(
        None, description="Force specific thinking modes, e.g. ['mode1', 'mode4']"
    )
    force_agent: str | None = Field(
        None, description="Force a specific agent, e.g. 'ARIEL', 'NYX', 'NOVA', 'ATLAS'"
    )
    max_modes: int = Field(2, ge=1, le=4, description="Max parallel thinking modes")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What fields are required for a Sales Invoice?",
                "force_modes": None,
                "max_modes": 2,
            }
        }


class MemoryRecordRequest(BaseModel):
    query: str
    response: str
    mode_used: str = "manual"
    tags: list[str] = Field(default_factory=list)


class VoiceRequest(BaseModel):
    text: str = Field(..., description="Text to process through the voice pipeline")


class BenchmarkRequest(BaseModel):
    concurrency: int = Field(5, ge=1, le=50, description="Concurrent requests")
    n_requests: int  = Field(10, ge=1, le=200, description="Total requests")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class AgentResultResponse(BaseModel):
    agent_name: str
    confidence: float
    latency_ms: float
    traceable: bool
    output_preview: str   # first 200 chars
    error: str = ""


class ModeResultResponse(BaseModel):
    mode: str
    confidence: float
    latency_ms: float
    traceable: bool
    error: str = ""


class QueryResponse(BaseModel):
    query: str
    primary_output: str
    output_type: str
    confidence: float
    modes_activated: list[str]
    routed_to_agent: str
    total_latency_ms: float
    memory_context_summary: str
    mode_results: list[ModeResultResponse]
    agent_results: list[AgentResultResponse]

    @classmethod
    def from_cognition_result(cls, result) -> "QueryResponse":
        return cls(
            query=result.query,
            primary_output=result.primary_output,
            output_type=result.output_type,
            confidence=result.confidence,
            modes_activated=[m.value for m in result.modes_activated],
            routed_to_agent=result.routed_to_agent,
            total_latency_ms=result.total_latency_ms,
            memory_context_summary=result.memory_context.get("summary", ""),
            mode_results=[
                ModeResultResponse(
                    mode=r.mode.value,
                    confidence=r.confidence,
                    latency_ms=r.latency_ms,
                    traceable=r.traceable,
                    error=r.error,
                )
                for r in result.mode_results
            ],
            agent_results=[
                AgentResultResponse(
                    agent_name=a.agent_name,
                    confidence=a.confidence,
                    latency_ms=a.latency_ms,
                    traceable=a.traceable,
                    output_preview=a.output[:200],
                    error=a.error,
                )
                for a in result.agent_results
            ],
        )


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agents_loaded: list[str]
    memory_stats: dict[str, Any]
    idle_loop_active: bool
    version: str = "0.1.0"


class AgentInfo(BaseModel):
    name: str
    description: str
    keywords: list[str]
    base_confidence: float


class AgentsResponse(BaseModel):
    agents: list[AgentInfo]
    count: int


class KGSearchResult(BaseModel):
    name: str
    score: float
    module: str
    is_submittable: bool


class KGSearchResponse(BaseModel):
    query: str
    results: list[KGSearchResult]
    total: int


class MemoryContextResponse(BaseModel):
    query: str
    summary: str
    recent_count: int
    similar_count: int
    semantic_facts_count: int
    working_slots: int
    top_similar: list[dict]
    top_facts: list[dict]


class VoiceResponse(BaseModel):
    transcription: str
    cognition_output: str
    synthesis_path: str
    total_latency_ms: float
    within_target: bool


class BenchmarkResponse(BaseModel):
    concurrency: int
    n_requests: int
    n_success: int
    error_rate: float
    p50_ms: float
    p99_ms: float
    throughput_rps: float
    passes_criterion: bool
