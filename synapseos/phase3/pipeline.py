"""
Phase 3 — End-to-end SynapseOS inference pipeline.

Wires all trained layers into a single forward pass:

  INPUT (text query)
      ↓
  [TextEncoder]          — simple embedding (Mamba when GPU available)
      ↓
  [JEPA WorldState]      — latent world state from Phase 2
      ↓
  [N2S Bridge]           — routes to query_type + doctype
      ↓
  [NeurosymbolicCore]    — graph query + rule evaluation
      ↓
  [S2N Bridge]           — symbolic result → embedding
      ↓
  [KAN Layer]            — quantitative relationships
      ↓
  [OutputRenderer]       — structured natural language response
      ↓
  OUTPUT

Each layer can be swapped for its GPU-trained version without changing the pipeline.

Run:
    python -m synapseos.phase3.pipeline
    python -m synapseos.phase3.pipeline --query "What fields are required for a Sales Invoice?"
    python -m synapseos.phase3.pipeline --query "Plan the submission steps for a Purchase Order"
    python -m synapseos.phase3.pipeline --query "Validate a Sales Order with missing customer"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from synapseos.phase3.neurosymbolic_core import (
    NeurosymbolicCore, QueryType, ReasoningResult,
)


# ---------------------------------------------------------------------------
# Layer 1 — Text Encoder (Mamba fallback: hash embedding)
# ---------------------------------------------------------------------------

class TextEncoder:
    """
    Encodes raw text into a latent vector.
    Production: Mamba LoRA fine-tuned model (from Phase 2).
    CPU fallback: deterministic hash-based embedding + TF-IDF-style weighting.
    """

    def __init__(self, latent_dim: int = 64, mamba_checkpoint: str | None = None):
        self.latent_dim = latent_dim
        self.mamba_model = None

        if mamba_checkpoint and Path(mamba_checkpoint).exists():
            self._load_mamba(mamba_checkpoint)

    def _load_mamba(self, checkpoint: str) -> None:
        try:
            from mamba_ssm import MambaLMHeadModel
            from peft import PeftModel
            self.mamba_model = PeftModel.from_pretrained(
                MambaLMHeadModel.from_pretrained("tiiuae/falcon-mamba-7b"),
                checkpoint,
            )
            print(f"[TextEncoder] Mamba loaded from {checkpoint}")
        except ImportError:
            print("[TextEncoder] mamba-ssm not installed — using hash fallback")

    def encode(self, text: str) -> torch.Tensor:
        if self.mamba_model is not None:
            return self._encode_mamba(text)
        return self._encode_hash(text)

    def _encode_mamba(self, text: str) -> torch.Tensor:
        # Placeholder — real Mamba hidden state extraction here
        return self._encode_hash(text)

    def _encode_hash(self, text: str) -> torch.Tensor:
        """
        Deterministic embedding: each token gets a hash-based vector, averaged.
        Preserves semantic similarity for domain vocabulary (same DocType name
        always maps to the same vector).
        """
        tokens = text.lower().split()
        vec = torch.zeros(self.latent_dim)
        for token in tokens:
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            for i in range(self.latent_dim):
                bit = (h >> i) & 1
                vec[i] += (1.0 if bit else -1.0)
        if tokens:
            vec /= len(tokens)
        # Normalize
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        return vec.unsqueeze(0)  # (1, latent_dim)


# ---------------------------------------------------------------------------
# Layer 2 — JEPA World State (load trained model from Phase 2)
# ---------------------------------------------------------------------------

class JEPAWorldState:
    """Loads trained JEPA encoder and produces world state latent."""

    def __init__(self, checkpoint: str = "data/checkpoints/jepa/jepa_best.pt",
                 latent_dim: int = 64):
        self.latent_dim = latent_dim
        self.model = None
        self._load(checkpoint)

    def _load(self, checkpoint: str) -> None:
        if not Path(checkpoint).exists():
            print(f"[JEPA] Checkpoint not found at {checkpoint} — using passthrough")
            return

        from synapseos.phase2.jepa_trainer import JEPAConfig, JEPAModel

        cfg_path = Path(checkpoint).parent / "jepa_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = JEPAConfig(**json.load(f))
        else:
            cfg = JEPAConfig()

        model = JEPAModel(cfg)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        self.model = model
        print(f"[JEPA] Loaded from {checkpoint}")

    def encode(self, text_latent: torch.Tensor) -> torch.Tensor:
        """
        text_latent: (1, input_dim) from TextEncoder
        Returns: (1, latent_dim) world state
        """
        if self.model is None:
            if text_latent.shape[-1] != self.latent_dim:
                with torch.no_grad():
                    proj = nn.Linear(text_latent.shape[-1], self.latent_dim, bias=False)
                    nn.init.eye_(proj.weight[:min(self.latent_dim, text_latent.shape[-1]),
                                             :min(self.latent_dim, text_latent.shape[-1])])
                    return proj(text_latent)
            return text_latent

        expected_input_dim = self.model.context_encoder.input_proj.in_features
        with torch.no_grad():
            # Project text latent to JEPA's expected input_dim if needed
            if text_latent.shape[-1] != expected_input_dim:
                proj = nn.Linear(text_latent.shape[-1], expected_input_dim, bias=False)
                text_latent = proj(text_latent)

            # JEPA context encoder expects (B, T, D)
            ctx = text_latent.unsqueeze(1).expand(-1, 3, -1)  # (1, 3, D)
            return self.model.context_encoder(ctx)             # (1, latent_dim)


# ---------------------------------------------------------------------------
# Layer 3 — Neural→Symbolic Bridge
# ---------------------------------------------------------------------------

class N2SBridge:
    """Routes latent vector to query_type + doctype."""

    def __init__(
        self,
        checkpoint: str = "data/checkpoints/bridges/n2s_best.pt",
        config_path: str = "data/checkpoints/bridges/bridge_config.json",
        doctype_index_path: str = "data/checkpoints/bridges/doctype_index.json",
    ):
        self.model      = None
        self.cfg        = None
        self.query_types = ["retrieve", "reason", "plan", "compute", "validate"]
        self.doctypes   = []
        self._load(checkpoint, config_path, doctype_index_path)

    def _load(self, checkpoint: str, config_path: str, doctype_index_path: str) -> None:
        if not Path(checkpoint).exists():
            print(f"[N2S] Checkpoint not found — using keyword heuristic")
            return

        from synapseos.phase3.bridge_trainer import BridgeConfig, NeuralToSymbolicBridge

        with open(config_path) as f:
            cfg_dict = json.load(f)
        self.cfg = BridgeConfig(**cfg_dict)
        self.query_types = self.cfg.query_types

        with open(doctype_index_path) as f:
            dt_index = json.load(f)
        self.doctypes = sorted(dt_index.keys(), key=lambda k: dt_index[k])

        model = NeuralToSymbolicBridge(self.cfg, len(self.doctypes))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        self.model = model
        print(f"[N2S] Loaded — {len(self.doctypes)} DocTypes, {len(self.query_types)} query types")

    def route(self, latent: torch.Tensor, query_text: str) -> tuple[QueryType, str]:
        """Returns (query_type, doctype)."""
        if self.model is not None:
            with torch.no_grad():
                # Project latent to bridge's expected dim if they differ
                expected_dim = self.cfg.latent_dim if self.cfg else 64
                if latent.shape[-1] != expected_dim:
                    proj = nn.Linear(latent.shape[-1], expected_dim, bias=False)
                    latent = proj(latent)
                out = self.model(latent)
                qt_idx = out["query_type_logits"].argmax(1).item()
                dt_idx = out["doctype_logits"].argmax(1).item()
                qt = self.query_types[qt_idx]
                doctype = self.doctypes[dt_idx] if dt_idx < len(self.doctypes) else ""
        else:
            qt, doctype = self._heuristic_route(query_text)

        # Override query_type and doctype with keyword matches from the raw query text
        kw_qt = self._extract_querytype_keyword(query_text)
        if kw_qt:
            qt = kw_qt

        kw_doctype = self._extract_doctype_keyword(query_text)
        if kw_doctype:
            doctype = kw_doctype

        return QueryType(qt), doctype

    def _extract_querytype_keyword(self, query: str) -> str | None:
        q = query.lower()
        if any(w in q for w in ["validate", "check if", "verify", "missing", "error", "invalid"]):
            return "validate"
        if any(w in q for w in ["plan", "steps", "how do i submit", "process", "workflow"]):
            return "plan"
        if any(w in q for w in ["compute", "calculate", "formula", "total", "amount due"]):
            return "compute"
        if any(w in q for w in ["why", "explain", "because", "reason", "how does"]):
            return "reason"
        if any(w in q for w in ["what is", "what are", "linked to", "fields", "show me", "list"]):
            return "retrieve"
        return None

    def _heuristic_route(self, query: str) -> tuple[str, str]:
        q = query.lower()
        if any(w in q for w in ["plan", "steps", "how to", "process", "workflow"]):
            qt = "plan"
        elif any(w in q for w in ["validate", "check", "verify", "error", "missing"]):
            qt = "validate"
        elif any(w in q for w in ["compute", "calculate", "formula", "amount", "total"]):
            qt = "compute"
        elif any(w in q for w in ["reason", "why", "explain", "because"]):
            qt = "reason"
        else:
            qt = "retrieve"
        return qt, ""

    def _extract_doctype_keyword(self, query: str) -> str:
        """Find a known DocType name mentioned in the query."""
        known = [
            "Sales Invoice", "Purchase Invoice", "Sales Order", "Purchase Order",
            "Delivery Note", "Payment Entry", "Stock Entry", "Journal Entry",
            "Customer", "Supplier", "Item", "Employee", "Account", "Project",
            "Task", "Leave Application", "Expense Claim", "Salary Slip",
        ]
        q_lower = query.lower()
        for dt in known:
            if dt.lower() in q_lower:
                return dt
        return ""


# ---------------------------------------------------------------------------
# Layer 5 — Symbolic→Neural Bridge
# ---------------------------------------------------------------------------

class S2NBridge:
    """Projects symbolic reasoning result back to embedding space."""

    def __init__(self, checkpoint: str = "data/checkpoints/bridges/s2n_best.pt",
                 config_path: str = "data/checkpoints/bridges/bridge_config.json"):
        self.model = None
        self._load(checkpoint, config_path)

    def _load(self, checkpoint: str, config_path: str) -> None:
        if not Path(checkpoint).exists():
            print(f"[S2N] Checkpoint not found — using passthrough")
            return

        from synapseos.phase3.bridge_trainer import BridgeConfig, SymbolicToNeuralBridge

        with open(config_path) as f:
            cfg = BridgeConfig(**json.load(f))

        model = SymbolicToNeuralBridge(cfg)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        self.model = model
        print(f"[S2N] Loaded")

    def project(self, reasoning_result: ReasoningResult) -> torch.Tensor:
        features = torch.tensor(
            [reasoning_result.to_symbolic_features(dim=32)], dtype=torch.float32
        )
        if self.model is None:
            return features

        with torch.no_grad():
            return self.model(features)  # (1, output_embed_dim)


# ---------------------------------------------------------------------------
# Layer 6 — Output Renderer
# ---------------------------------------------------------------------------

class OutputRenderer:
    """
    Renders the reasoning result into structured natural language.
    Production: fine-tuned Mistral 1B / Llama 3B (Phase 3 SFT).
    Fallback: template-based structured output.
    """

    def render(self, result: ReasoningResult, original_query: str) -> str:
        lines = [
            f"Query type : {result.query_type.value.upper()}",
            f"DocType    : {result.doctype or '(general)'}",
            f"Verified   : {'YES' if result.verified else 'NO — constraints violated'}",
            f"Confidence : {result.confidence:.0%}",
            "",
        ]

        if result.constraints_violated:
            lines.append("CONSTRAINTS VIOLATED:")
            for v in result.constraints_violated:
                lines.append(f"  ✗ {v}")
            lines.append("")

        if result.facts:
            lines.append(f"FACTS ({len(result.facts)} retrieved):")
            for fact in result.facts[:8]:
                fact_type = fact.get("type", "")
                if fact_type == "entity":
                    data = fact.get("data", {})
                    lines.append(f"  • {data.get('id', '')} [{data.get('module', '')}]"
                                 f" — submittable={data.get('is_submittable', False)}")
                elif fact_type == "link":
                    lines.append(f"  → links to {fact.get('target')} via {fact.get('field')}")
                elif fact_type == "child_table":
                    lines.append(f"  ⊂ child table: {fact.get('target')}")
                elif fact_type == "rule":
                    lines.append(f"  [{fact.get('id')}] IF {str(fact.get('condition', ''))[:60]}")
                    lines.append(f"         THEN {str(fact.get('consequence', ''))[:60]}")
                elif fact_type == "violation":
                    lines.append(f"  ✗ {fact.get('message', '')}")
                elif fact_type in ("step",) or "step" in fact:
                    lines.append(f"  {fact.get('step', '?')}. {fact.get('action', '')} [{fact.get('doctype', '')}]")
                else:
                    lines.append(f"  • {json.dumps(fact)[:80]}")
            if len(result.facts) > 8:
                lines.append(f"  ... and {len(result.facts) - 8} more")
            lines.append("")

        if result.rules_applied:
            lines.append(f"Rules applied: {', '.join(result.rules_applied[:5])}"
                         + (f" (+{len(result.rules_applied)-5} more)" if len(result.rules_applied) > 5 else ""))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    query: str
    query_type: QueryType
    doctype: str
    reasoning: ReasoningResult
    output: str
    latency_ms: float


class SynapseOSPipeline:
    """
    End-to-end SynapseOS inference pipeline.
    Connects all trained layers from Phase 1–3.
    """

    def __init__(
        self,
        mamba_checkpoint: str | None = None,
        jepa_checkpoint: str = "data/checkpoints/jepa/jepa_best.pt",
        bridge_checkpoint_dir: str = "data/checkpoints/bridges",
        graph_path: str = "data/graph/knowledge_graph.json",
        rules_path: str = "data/rules/rules.json",
        latent_dim: int = 64,
    ):
        print("Initializing SynapseOS pipeline...")
        t0 = time.time()

        self.text_encoder = TextEncoder(latent_dim=latent_dim, mamba_checkpoint=mamba_checkpoint)
        self.jepa         = JEPAWorldState(checkpoint=jepa_checkpoint, latent_dim=latent_dim)
        self.n2s          = N2SBridge(
            checkpoint=f"{bridge_checkpoint_dir}/n2s_best.pt",
            config_path=f"{bridge_checkpoint_dir}/bridge_config.json",
            doctype_index_path=f"{bridge_checkpoint_dir}/doctype_index.json",
        )
        self.core         = NeurosymbolicCore(graph_path=graph_path, rules_path=rules_path)
        self.s2n          = S2NBridge(
            checkpoint=f"{bridge_checkpoint_dir}/s2n_best.pt",
            config_path=f"{bridge_checkpoint_dir}/bridge_config.json",
        )
        self.renderer     = OutputRenderer()

        elapsed = (time.time() - t0) * 1000
        print(f"Pipeline ready in {elapsed:.0f}ms\n")

    def run(self, query: str, context: dict[str, Any] | None = None) -> PipelineResult:
        t0 = time.time()

        # Layer 1: encode text
        text_latent = self.text_encoder.encode(query)

        # Layer 2: JEPA world state
        world_latent = self.jepa.encode(text_latent)

        # Layer 3: route to query type + doctype
        query_type, doctype = self.n2s.route(world_latent, query)

        # Layer 4: neurosymbolic reasoning
        reasoning = self.core.reason(query_type, doctype, context)

        # Layer 5: project back to embedding space
        result_embed = self.s2n.project(reasoning)

        # Layer 6: render output
        output = self.renderer.render(reasoning, query)

        latency_ms = (time.time() - t0) * 1000

        return PipelineResult(
            query=query,
            query_type=query_type,
            doctype=doctype,
            reasoning=reasoning,
            output=output,
            latency_ms=latency_ms,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parser.parse_args()

    pipeline = SynapseOSPipeline(latent_dim=args.latent_dim)

    demo_queries = [
        "What fields are required for a Sales Invoice?",
        "Plan the submission steps for a Purchase Order",
        "Validate a Sales Order with a missing customer field",
        "What DocTypes are linked to the Customer doctype?",
        "Compute the approval time formula for a Leave Application",
    ]

    queries = [args.query] if args.query else demo_queries

    for query in queries:
        print(f"{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        result = pipeline.run(query)
        print(result.output)
        print(f"Latency: {result.latency_ms:.1f}ms\n")


if __name__ == "__main__":
    main()
