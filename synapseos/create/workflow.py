"""
Create — node-graph workflow engine (ComfyUI's idea, SynapseOS's body).

A creative pipeline is a JSON DAG: nodes keyed by id, each with a type and
inputs; an input may reference another node's output as ["node_id", "key"].
The engine resolves dependencies topologically and executes — chaining the
native generators instead of diffusion checkpoints:

    {
      "1": {"type": "lyrics.write",  "inputs": {"theme": "the river", "emotion": "calm"}},
      "2": {"type": "song.sing",     "inputs": {"theme": "the river", "raag": "bhupali"}},
      "3": {"type": "video.create",  "inputs": {"concept": "flowing river",
                                                "soundtrack": ["2", "path"]}},
      "4": {"type": "poster.create", "inputs": {"title": ["1", "title"],
                                                "subtitle": "an original song"}}
    }

Every run produces a manifest (node order, inputs after resolution, outputs,
timing) — the workflow itself is the provenance. Deterministic given seeds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class WorkflowResult:
    outputs: dict[str, dict]            # node_id → output dict
    order: list[str]
    duration_s: float
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"order": self.order, "duration_s": round(self.duration_s, 2),
                "outputs": self.outputs, "errors": self.errors}


# ---------------------------------------------------------------------------
# Node implementations — thin adapters over the native generators
# ---------------------------------------------------------------------------

def _node_music(inputs: dict) -> dict:
    from synapseos.create.music import MusicComposer
    comp = MusicComposer().compose(
        mood=inputs.get("mood", "calm"), bars=int(inputs.get("bars", 8)),
        seed=inputs.get("seed"))
    return comp.to_dict()


def _node_song(inputs: dict) -> dict:
    from synapseos.create.song import SongComposer
    song = SongComposer().sing(
        theme=inputs.get("theme", "the journey"), raag=inputs.get("raag"),
        emotion=inputs.get("emotion"), instrument=inputs.get("instrument", "sitar"),
        cycles=int(inputs.get("cycles", 4)), seed=inputs.get("seed"))
    return song.to_dict()


def _node_lyrics(inputs: dict) -> dict:
    from synapseos.create.lyrics import LyricWriter
    lyr = LyricWriter().write(inputs.get("theme", "life"),
                              emotion=inputs.get("emotion", "calm"),
                              seed=inputs.get("seed"))
    return {"title": lyr.title, "text": lyr.full_text(), "verse": lyr.verse,
            "chorus": lyr.chorus}


def _node_image(inputs: dict) -> dict:
    from synapseos.create.image import ImageGenerator
    img = ImageGenerator().create(
        inputs.get("concept", "abstract"), style=inputs.get("style", "auto"),
        width=int(inputs.get("width", 640)), height=int(inputs.get("height", 480)),
        seed=inputs.get("seed"))
    return img.to_dict()


def _node_video(inputs: dict) -> dict:
    from synapseos.create.video import VideoGenerator
    vid = VideoGenerator().create(
        inputs.get("concept", "abstract"), seconds=float(inputs.get("seconds", 4.0)),
        fps=int(inputs.get("fps", 12)), seed=inputs.get("seed"),
        soundtrack=inputs.get("soundtrack"))
    return vid.to_dict()


def _node_poster(inputs: dict) -> dict:
    from synapseos.create.poster import PosterGenerator, PosterSpec
    spec = PosterSpec(
        title=inputs.get("title", "untitled"),
        subtitle=inputs.get("subtitle", ""),
        body=list(inputs.get("body", [])),
        palette=inputs.get("palette"),
        background=inputs.get("background", "flow"),
        footer=inputs.get("footer", ""))
    poster = PosterGenerator().create(spec, seed=inputs.get("seed"))
    return poster.to_dict()


NODE_TYPES: dict[str, Callable[[dict], dict]] = {
    "music.compose": _node_music,
    "song.sing": _node_song,
    "lyrics.write": _node_lyrics,
    "image.create": _node_image,
    "video.create": _node_video,
    "poster.create": _node_poster,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """Resolve the DAG, execute nodes in dependency order, keep a manifest."""

    def __init__(self, manifest_dir: str = "data/creations/workflows"):
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_ref(value: Any) -> bool:
        return (isinstance(value, list) and len(value) == 2
                and isinstance(value[0], str) and isinstance(value[1], str))

    def _deps(self, node: dict, node_ids: set[str]) -> set[str]:
        deps = set()
        for v in node.get("inputs", {}).values():
            if self._is_ref(v) and v[0] in node_ids:
                deps.add(v[0])
        return deps

    def run(self, workflow: dict[str, dict]) -> WorkflowResult:
        t0 = time.time()
        ids = set(workflow)
        # Kahn's algorithm
        remaining = dict(workflow)
        order: list[str] = []
        while remaining:
            ready = [nid for nid, node in remaining.items()
                     if self._deps(node, ids) <= set(order)]
            if not ready:
                raise ValueError(f"workflow has a cycle or missing node: "
                                 f"{sorted(remaining)}")
            for nid in sorted(ready):
                order.append(nid)
                del remaining[nid]

        outputs: dict[str, dict] = {}
        errors: dict[str, str] = {}
        for nid in order:
            node = workflow[nid]
            ntype = node.get("type", "")
            fn = NODE_TYPES.get(ntype)
            if fn is None:
                errors[nid] = f"unknown node type '{ntype}'"
                continue
            # Resolve references against upstream outputs
            resolved = {}
            for key, value in node.get("inputs", {}).items():
                if self._is_ref(value) and value[0] in outputs:
                    resolved[key] = outputs[value[0]].get(value[1])
                else:
                    resolved[key] = value
            try:
                outputs[nid] = fn(resolved)
            except Exception as e:
                errors[nid] = f"{type(e).__name__}: {e}"

        result = WorkflowResult(outputs=outputs, order=order,
                                duration_s=time.time() - t0, errors=errors)
        manifest = self.manifest_dir / f"run_{int(t0)}.json"
        manifest.write_text(json.dumps(
            {"workflow": workflow, **result.to_dict()}, indent=2))
        return result
