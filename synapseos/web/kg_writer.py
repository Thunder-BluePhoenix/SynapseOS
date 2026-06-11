"""
Web Learning Loop — Knowledge Graph overlay writer.

The curated Frappe knowledge graph (data/graph/knowledge_graph.json) stays
read-only. Web-learned knowledge lives in a separate overlay file
(data/web/web_graph.json, same NetworkX node-link format) and is merged into
the in-memory KnowledgeGraph at load time, so search()/get_neighbors() see
both worlds while the curated file is never touched.

Every overlay node/edge carries provenance: {url, timestamp}.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class KGWriter:
    """Write + persist layer over the phase3 KnowledgeGraph for web knowledge."""

    def __init__(self, kg, graph_path: str = "data/web/web_graph.json"):
        self.kg = kg
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Track which nodes/edges are web-originated so save() only persists those.
        self._web_nodes: dict[str, dict[str, Any]] = {}
        self._web_edges: list[dict[str, Any]] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self._load_overlay()

    # -- persistence ----------------------------------------------------------

    def _load_overlay(self) -> None:
        if not self.graph_path.exists():
            return
        try:
            data = json.loads(self.graph_path.read_text())
        except Exception:
            return
        for node in data.get("nodes", []):
            nid = node.get("id", "")
            if nid:
                self._web_nodes[nid] = node
                self._merge_node(nid, node)
        for edge in data.get("links", data.get("edges", [])):
            key = (edge.get("source", ""), edge.get("relation", ""), edge.get("target", ""))
            if key[0] and key[2] and key not in self._edge_keys:
                self._edge_keys.add(key)
                self._web_edges.append(edge)
                self._merge_edge(edge)

    def save(self) -> None:
        with self._lock:
            data = {
                "directed": True,
                "nodes": list(self._web_nodes.values()),
                "links": self._web_edges,
            }
            self.graph_path.write_text(json.dumps(data, indent=2))

    # -- in-memory merge into the live KnowledgeGraph --------------------------

    def _merge_node(self, nid: str, node: dict[str, Any]) -> None:
        if nid not in self.kg._nodes:
            self.kg._nodes[nid] = node

    def _merge_edge(self, edge: dict[str, Any]) -> None:
        self.kg._edges.append(edge)
        self.kg._adjacency.setdefault(edge["source"], []).append(edge)

    # -- public API -------------------------------------------------------------

    def add_entity(self, name: str, node_type: str = "concept",
                   provenance: dict | None = None) -> bool:
        """Add a concept node. Returns False if it already exists anywhere."""
        with self._lock:
            if name in self.kg._nodes:
                return False
            node = {
                "id": name,
                "type": node_type,
                "source": "web",
                "provenance": provenance or {},
            }
            self._web_nodes[name] = node
            self._merge_node(name, node)
            return True

    def add_fact(self, subject: str, relation: str, obj: str,
                 confidence: float, url: str) -> bool:
        """Add a (subject, relation, obj) edge with provenance. Dedups."""
        with self._lock:
            key = (subject, relation, obj)
            if key in self._edge_keys:
                return False
            prov = {"url": url, "timestamp": time.time()}
            for name in (subject, obj):
                if name not in self.kg._nodes:
                    node = {"id": name, "type": "concept", "source": "web",
                            "provenance": prov}
                    self._web_nodes[name] = node
                    self._merge_node(name, node)
            edge = {
                "source": subject,
                "target": obj,
                "relation": relation,
                "confidence": confidence,
                "provenance": prov,
            }
            self._edge_keys.add(key)
            self._web_edges.append(edge)
            self._merge_edge(edge)
            return True

    def degrade_edge(self, subject: str, relation: str, obj: str,
                     confidence: float) -> bool:
        """Lower an edge's confidence after a belief revision. Returns found."""
        with self._lock:
            for edge in self._web_edges:
                if (edge.get("source") == subject and edge.get("relation") == relation
                        and edge.get("target") == obj):
                    edge["confidence"] = confidence
                    return True
            return False

    def stats(self) -> dict:
        return {"web_nodes": len(self._web_nodes), "web_edges": len(self._web_edges)}
