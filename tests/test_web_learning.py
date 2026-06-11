"""
Tests for the SynapseOS Web Learning Loop — curiosity-driven web ingestion.

Covers:
  - FetchPolicy: domain allowlist, session budget, scheme rejection
  - Seen-store persistence + FakeFetcher
  - TextExtractor: boilerplate/script stripping
  - FactExtractor: pattern triples, topic-hint pronoun resolution, caps
  - KGWriter: overlay add/persist/dedup, reload through public KnowledgeGraph
  - BeliefReviser: added / reinforced / revised / rejected / coexists
  - CuriosityEngine: hypotheses, KG frontier, user topics; GoalQueue ordering
  - WebLearningLoop: end-to-end with FakeFetcher + real temp WorldModel
  - CognitiveController integration: disabled by default, enabled + clean shutdown

NO network access — everything runs against FakeFetcher fixtures.

Run:
    python tests/test_web_learning.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.web.fetcher import (
    FakeFetcher, FetchPolicy, FetchResult, WebFetcher, url_hash,
)
from synapseos.web.extractor import ExtractedFact, FactExtractor, TextExtractor
from synapseos.web.kg_writer import KGWriter
from synapseos.web.belief import BeliefReviser, source_trust
from synapseos.web.curiosity import CuriosityEngine, GoalQueue, LearningGoal
from synapseos.web.learning_loop import WebLearningLoop
from synapseos.web.media import FakeMediaIngester, classify_media, _parse_subtitles
from synapseos.phase3.neurosymbolic_core import KnowledgeGraph
from synapseos.phase5.memory import SemanticMemory
from synapseos.phase4.idle_loop import HypothesisStore, WeakHypothesis


results = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                r = {"name": name, "passed": True,  "ms": round((time.time()-t0)*1000, 1), "detail": ""}
            except AssertionError as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": str(e)}
            except Exception as e:
                r = {"name": name, "passed": False, "ms": round((time.time()-t0)*1000, 1), "detail": f"{type(e).__name__}: {e}"}
            results.append(r)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html><head><title>Photosynthesis - Test Page</title>
<script>function noise() { return 42; }</script>
<style>.nav { color: red; }</style></head>
<body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<h1>Photosynthesis</h1>
<p>Photosynthesis is a biological process used by plants to convert light into energy.
Chlorophyll is a green pigment found in chloroplasts. The process was discovered by Jan Ingenhousz.</p>
<li>Short</li>
<footer>Copyright 2026 — all rights reserved by somebody somewhere</footer>
</body></html>
"""

WIKI_PHOTO = {
    "title": "Photosynthesis",
    "extract": ("Photosynthesis is a biological process. It is used by plants to make food. "
                "Chlorophyll is a green pigment. Photosynthesis was discovered by Jan Ingenhousz."),
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Photosynthesis"}},
}

WIKI_PLUTO = {
    "title": "Pluto",
    "extract": "Pluto is a dwarf planet. Pluto is located in the Kuiper belt.",
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Pluto"}},
}


def _tmp_dir() -> str:
    return tempfile.mkdtemp(prefix="web_test_")


def _tmp_semantic() -> SemanticMemory:
    d = _tmp_dir()
    return SemanticMemory(
        store_path=os.path.join(d, "semantic.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
    )


def _tmp_kg_writer() -> tuple[KGWriter, str]:
    d = _tmp_dir()
    kg = KnowledgeGraph(os.path.join(d, "missing.json"))
    path = os.path.join(d, "web_graph.json")
    return KGWriter(kg, graph_path=path), path


def _fact(s, p, o, conf=0.6) -> ExtractedFact:
    return ExtractedFact(subject=s, predicate=p, obj=o, confidence=conf,
                         sentence=f"{s} {p} {o}.", pattern=p)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

@test("fetch policy blocks disallowed domain")
def test_policy_domain():
    f = WebFetcher(policy=FetchPolicy(allowed_domains=["wikipedia.org"]),
                   seen_path=os.path.join(_tmp_dir(), "seen.json"))
    ok, reason = f._allowed("https://evil.example.com/page")
    assert not ok and "allowlist" in reason
    ok, _ = f._allowed("https://en.wikipedia.org/wiki/Cat")
    # robots check may be skipped only if policy allows; disable robots for offline test
    f2 = WebFetcher(policy=FetchPolicy(allowed_domains=["wikipedia.org"], respect_robots=False),
                    seen_path=os.path.join(_tmp_dir(), "seen.json"))
    ok2, _ = f2._allowed("https://en.wikipedia.org/wiki/Cat")
    assert ok2


@test("fetch policy rejects non-http schemes")
def test_policy_scheme():
    f = WebFetcher(policy=FetchPolicy(respect_robots=False),
                   seen_path=os.path.join(_tmp_dir(), "seen.json"))
    ok, reason = f._allowed("file:///etc/passwd")
    assert not ok and "scheme" in reason
    ok, reason = f._allowed("ftp://example.com/x")
    assert not ok


@test("fetch policy session budget blocks")
def test_policy_budget():
    f = WebFetcher(policy=FetchPolicy(max_pages_per_session=2, respect_robots=False),
                   seen_path=os.path.join(_tmp_dir(), "seen.json"))
    f._session_pages = 2
    result = f.fetch("https://example.com/page")
    assert result.status == -1 and "budget" in result.error


@test("seen store persists across instances")
def test_seen_persist():
    d = _tmp_dir()
    path = os.path.join(d, "seen.json")
    f = WebFetcher(seen_path=path)
    f.mark_seen("https://example.com/a", 200)
    f.save()
    f2 = WebFetcher(seen_path=path)
    assert f2.seen("https://example.com/a")
    assert not f2.seen("https://example.com/b")


@test("url hash canonicalizes")
def test_url_hash():
    assert url_hash("https://Example.com/page/") == url_hash("https://example.com/page")
    assert url_hash("https://example.com/a") != url_hash("https://example.com/b")


@test("fake fetcher returns canned pages and records calls")
def test_fake_fetcher():
    ff = FakeFetcher(pages={"https://x.com/p": "<p>hello world from fixture</p>"},
                     wiki={"Pluto": WIKI_PLUTO})
    r = ff.fetch("https://x.com/p")
    assert r.ok and "hello world" in r.text
    r404 = ff.fetch("https://x.com/missing")
    assert not r404.ok
    w = ff.wiki_summary("Pluto")
    assert w.ok and "dwarf planet" in w.text
    assert ("fetch", "https://x.com/p") in ff.calls


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

@test("text extractor strips scripts and boilerplate")
def test_text_extract():
    text = TextExtractor().extract(SAMPLE_HTML)
    assert "function noise" not in text
    assert ".nav" not in text
    assert "Photosynthesis is a biological process" in text
    assert "Home" not in text          # nav stripped
    assert "Short" not in text         # below min block length


@test("text extractor gets title")
def test_text_title():
    title = TextExtractor().extract_title(SAMPLE_HTML)
    assert "Photosynthesis" in title


@test("fact extractor is_a and created_by patterns")
def test_fact_patterns():
    text = ("Photosynthesis is a biological process. "
            "Python was created by Guido van Rossum. "
            "Paris is located in France.")
    facts = FactExtractor().extract(text)
    triples = {(f.subject, f.predicate) for f in facts}
    assert ("Photosynthesis", "is_a") in triples
    assert ("Python", "created_by") in triples
    assert ("Paris", "located_in") in triples
    for f in facts:
        assert f.sentence  # provenance kept


@test("fact extractor topic hint resolves pronoun")
def test_fact_pronoun():
    facts = FactExtractor().extract("It is a biological process.",
                                    topic_hint="photosynthesis")
    assert any(f.subject == "Photosynthesis" and f.predicate == "is_a" for f in facts)


@test("fact extractor caps facts per page and dedups")
def test_fact_caps():
    text = " ".join(f"Thing{i} is a widget." for i in range(100))
    fx = FactExtractor(max_facts_per_page=10)
    facts = fx.extract(text)
    assert len(facts) <= 10
    dup = FactExtractor().extract("Pluto is a dwarf planet. Pluto is a dwarf planet.")
    assert len(dup) == 1


# ---------------------------------------------------------------------------
# KG writer
# ---------------------------------------------------------------------------

@test("kg writer adds facts and persists overlay")
def test_kg_writer_persist():
    writer, path = _tmp_kg_writer()
    assert writer.add_fact("Pluto", "is_a", "dwarf planet", 0.6, "https://w.org/Pluto")
    writer.save()
    # Reload through the PUBLIC KnowledgeGraph loader — format-drift guard
    kg2 = KnowledgeGraph(path)
    assert kg2.get_entity("Pluto") is not None
    neigh = kg2.get_neighbors("Pluto")
    assert any(e["target"] == "dwarf planet" for e in neigh)


@test("kg writer dedups edges and merges in-memory")
def test_kg_writer_dedup():
    writer, _ = _tmp_kg_writer()
    assert writer.add_fact("A", "is_a", "B", 0.6, "u")
    assert not writer.add_fact("A", "is_a", "B", 0.6, "u")
    # live KG sees the merged knowledge immediately
    assert writer.kg.get_entity("A") is not None
    assert len(writer.kg.get_neighbors("A")) == 1


@test("kg writer overlay reloads on init")
def test_kg_writer_reload():
    writer, path = _tmp_kg_writer()
    writer.add_fact("X", "part_of", "Y", 0.6, "u")
    writer.save()
    kg = KnowledgeGraph(os.path.join(_tmp_dir(), "missing.json"))
    writer2 = KGWriter(kg, graph_path=path)
    assert kg.get_entity("X") is not None
    assert writer2.stats()["web_edges"] == 1


# ---------------------------------------------------------------------------
# Belief revision
# ---------------------------------------------------------------------------

@test("belief added when no conflict")
def test_belief_added():
    mem = _tmp_semantic()
    rev = BeliefReviser(mem, audit_path=os.path.join(_tmp_dir(), "rev.json"))
    r = rev.integrate(_fact("Pluto", "is_a", "dwarf planet"),
                      "https://en.wikipedia.org/wiki/Pluto")
    assert r.action == "added"
    facts = mem.retrieve("Pluto")
    assert any(f.obj == "dwarf planet" and f.source == "web:en.wikipedia.org"
               for f in facts)


@test("belief reinforced on duplicate")
def test_belief_reinforced():
    mem = _tmp_semantic()
    rev = BeliefReviser(mem, audit_path=os.path.join(_tmp_dir(), "rev.json"))
    rev.integrate(_fact("Pluto", "is_a", "dwarf planet"), "https://a.org/1")
    r = rev.integrate(_fact("Pluto", "is_a", "dwarf planet"), "https://b.org/2")
    assert r.action == "reinforced"
    assert r.new_confidence > r.old_confidence


@test("belief revised on stronger evidence")
def test_belief_revised():
    mem = _tmp_semantic()
    # Seed a weak old belief from a low-trust source
    mem.add_fact("Pluto", "is_a", "planet", confidence=0.3, source="inferred")
    # Age it so recency decay weakens it further
    mem.facts[-1].timestamp = "2024-01-01T00:00:00"
    rev = BeliefReviser(mem, audit_path=os.path.join(_tmp_dir(), "rev.json"))
    r = rev.integrate(_fact("Pluto", "is_a", "dwarf planet", conf=0.6),
                      "https://en.wikipedia.org/wiki/Pluto")
    assert r.action == "revised", f"got {r.action}: {r.reason}"
    # Old belief degraded but kept
    old = [f for f in mem.retrieve("Pluto") if f.obj == "planet"]
    assert old and old[0].confidence == 0.2


@test("belief rejected when existing belief stronger")
def test_belief_rejected():
    mem = _tmp_semantic()
    mem.add_fact("Pluto", "is_a", "dwarf planet", confidence=0.95, source="user_stated")
    rev = BeliefReviser(mem, audit_path=os.path.join(_tmp_dir(), "rev.json"))
    r = rev.integrate(_fact("Pluto", "is_a", "planet", conf=0.5),
                      "https://randomsite.example/blog")
    assert r.action == "rejected", f"got {r.action}: {r.reason}"
    # Not stored
    assert not any(f.obj == "planet" for f in mem.retrieve("Pluto"))


@test("belief coexists for non-functional predicate")
def test_belief_coexists():
    mem = _tmp_semantic()
    rev = BeliefReviser(mem, audit_path=os.path.join(_tmp_dir(), "rev.json"))
    rev.integrate(_fact("Water", "has_property", "high heat capacity"), "https://a.org")
    r = rev.integrate(_fact("Water", "has_property", "surface tension"), "https://b.org")
    assert r.action == "coexists"
    objs = {f.obj for f in mem.retrieve("Water")}
    assert {"high heat capacity", "surface tension"} <= objs


@test("belief audit log persists and trust ordering sane")
def test_belief_audit():
    d = _tmp_dir()
    mem = _tmp_semantic()
    path = os.path.join(d, "rev.json")
    rev = BeliefReviser(mem, audit_path=path)
    rev.integrate(_fact("A", "is_a", "B"), "https://x.org")
    rev.save()
    rev2 = BeliefReviser(mem, audit_path=path)
    assert len(rev2.revisions()) == 1
    assert source_trust("user_stated") > source_trust("web:en.wikipedia.org") > \
           source_trust("web:unknown-blog.example")


# ---------------------------------------------------------------------------
# Curiosity
# ---------------------------------------------------------------------------

@test("goal queue dedups, orders by priority, persists")
def test_goal_queue():
    d = _tmp_dir()
    q = GoalQueue(os.path.join(d, "goals.json"))
    g1 = LearningGoal(id="a", topic="low", reason="", source="kg_frontier", priority=0.3)
    g2 = LearningGoal(id="b", topic="high", reason="", source="user", priority=1.0)
    assert q.add(g1) and q.add(g2)
    assert not q.add(g1)                      # dedup
    assert q.pop().topic == "high"            # priority order
    q.mark("b", "done")
    assert q.pop().topic == "low"
    q.save()
    q2 = GoalQueue(os.path.join(d, "goals.json"))
    assert len(q2) == 2


@test("curiosity from weak hypotheses")
def test_curiosity_hypotheses():
    store = HypothesisStore(os.path.join(_tmp_dir(), "hyp.json"))
    store.add(WeakHypothesis(
        id="", type="cross_domain_link", concept_a="Mitochondria", concept_b="Battery",
        description="both store energy", confidence=0.3, observed_at="now"))
    engine = CuriosityEngine(hypothesis_store=store,
                             queue=GoalQueue(os.path.join(_tmp_dir(), "g.json")))
    goals = engine.generate_goals()
    assert any(g.source == "hypothesis" and "Mitochondria" in g.topic for g in goals)


@test("curiosity from kg frontier")
def test_curiosity_frontier():
    writer, path = _tmp_kg_writer()
    writer.add_fact("Quantum Computing", "is_a", "computing paradigm", 0.6, "u")
    engine = CuriosityEngine(knowledge_graph=writer.kg,
                             queue=GoalQueue(os.path.join(_tmp_dir(), "g.json")))
    goals = engine.generate_goals()
    assert any(g.source == "kg_frontier" for g in goals)


@test("user topic gets highest priority")
def test_curiosity_user():
    engine = CuriosityEngine(queue=GoalQueue(os.path.join(_tmp_dir(), "g.json")))
    g = engine.add_user_topic("Photosynthesis")
    assert g.priority == 1.0 and g.source == "user"
    assert engine.queue.pop().topic == "Photosynthesis"


# ---------------------------------------------------------------------------
# Learning loop end-to-end
# ---------------------------------------------------------------------------

def _build_loop(with_world_model: bool = True):
    d = _tmp_dir()
    ff = FakeFetcher(
        wiki={"Photosynthesis": WIKI_PHOTO, "Pluto": WIKI_PLUTO},
        searches={"Photosynthesis": ["Photosynthesis"], "Pluto": ["Pluto"]},
    )
    mem = _tmp_semantic()
    kg = KnowledgeGraph(os.path.join(d, "missing.json"))
    writer = KGWriter(kg, graph_path=os.path.join(d, "web_graph.json"))
    reviser = BeliefReviser(mem, kg_writer=writer,
                            audit_path=os.path.join(d, "rev.json"))
    wm = None
    if with_world_model:
        from synapseos.world_model.world_model import WorldModel
        wm = WorldModel(
            state_path=os.path.join(d, "state.json"),
            episodes_path=os.path.join(d, "episodes.json"),
            snapshot_dir=os.path.join(d, "snaps"),
        )
    curiosity = CuriosityEngine(world_model=wm, knowledge_graph=kg,
                                queue=GoalQueue(os.path.join(d, "goals.json")))
    loop = WebLearningLoop(fetcher=ff, curiosity=curiosity, reviser=reviser,
                           world_model=wm, data_dir=d)
    return loop, mem, kg, d


@test("learning loop end to end with fake fetcher")
def test_loop_end_to_end():
    loop, mem, kg, d = _build_loop()
    report = loop.learn("Photosynthesis")
    assert report.urls_fetched, f"errors: {report.errors}"
    assert report.facts_extracted > 0
    assert report.facts_added > 0
    # Knowledge landed in semantic memory with web provenance
    facts = mem.retrieve("Photosynthesis")
    assert any(f.source.startswith("web:") for f in facts)
    # World model perceived the page (episodes tagged "web")
    wm_eps = loop.world_model.memory.episodes
    assert any("web" in ep.tags for ep in wm_eps)
    # Goal marked done + session log written
    assert loop.curiosity.queue.stats()["by_status"].get("done", 0) >= 1
    assert (Path(d) / "session_log.json").exists()


@test("learning loop run batch respects n_goals")
def test_loop_batch():
    loop, _, _, _ = _build_loop(with_world_model=False)
    loop.curiosity.add_user_topic("Photosynthesis")
    loop.curiosity.add_user_topic("Pluto")
    reports = loop.run(n_goals=2)
    assert len(reports) == 2
    topics = {r.topic for r in reports}
    assert topics == {"Photosynthesis", "Pluto"}


@test("learning loop handles unknown topic gracefully")
def test_loop_unknown_topic():
    loop, _, _, _ = _build_loop(with_world_model=False)
    report = loop.learn("Completely Unknown Zorbloid Topic")
    assert report.facts_added == 0
    assert report.errors  # no sources found


@test("learning loop background start stop")
def test_loop_background():
    loop, _, _, _ = _build_loop(with_world_model=False)
    loop.idle_threshold = 0.0   # immediately idle
    loop.curiosity.add_user_topic("Pluto")
    loop.start_background(interval=0.05)
    time.sleep(0.4)
    loop.stop()
    assert loop._thread is None
    assert loop.stats()["goals_pursued"] >= 1


# ---------------------------------------------------------------------------
# Media ingestion
# ---------------------------------------------------------------------------

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
Mitochondria are the powerhouse of the cell.

2
00:00:04,000 --> 00:00:08,000
Mitochondria are the powerhouse of the cell.

3
00:00:08,000 --> 00:00:12,000
They produce energy through respiration.
"""


@test("media classification by url")
def test_media_classify():
    assert classify_media("https://www.youtube.com/watch?v=abc123") == "youtube"
    assert classify_media("https://youtu.be/abc123") == "youtube"
    assert classify_media("https://example.com/podcast/episode.mp3") == "audio"
    assert classify_media("https://example.com/talk.mp4") == "video"
    assert classify_media("https://example.com/article.html") is None
    assert classify_media("https://en.wikipedia.org/wiki/Cat") is None


@test("subtitle parser strips timestamps and dedups repeats")
def test_subtitle_parse():
    text = _parse_subtitles(SAMPLE_SRT)
    assert "powerhouse of the cell" in text
    assert "-->" not in text and "00:00" not in text
    assert text.count("powerhouse") == 1     # repeated caption deduped
    assert "respiration" in text


@test("learning loop ingests youtube via media ingester")
def test_loop_media_youtube():
    loop, mem, _, _ = _build_loop(with_world_model=False)
    yt_url = "https://www.youtube.com/watch?v=cellbio42"
    loop.media = FakeMediaIngester(transcripts={
        yt_url: ("Mitochondria are the powerhouse of the cell. "
                 "Respiration is a chemical process."),
    })
    report = loop.learn_url(yt_url, topic="cell biology")
    assert report.urls_fetched == [yt_url], f"errors: {report.errors}"
    assert report.facts_added > 0
    facts = mem.retrieve("Mitochondria")
    assert any("powerhouse" in f.obj for f in facts)
    assert loop.media.calls == [yt_url]


@test("learning loop reports media failure gracefully")
def test_loop_media_failure():
    loop, _, _, _ = _build_loop(with_world_model=False)
    loop.media = FakeMediaIngester()    # empty fixture set
    report = loop.learn_url("https://youtu.be/unknown999")
    assert report.facts_added == 0
    assert any("media ingest failed" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

@test("controller web learning disabled by default")
def test_controller_disabled():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = _tmp_dir()
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_agents=False, enable_world_model=False,
    )
    assert c.web_loop is None
    assert c.stats()["web_learning"] is None
    res = c.learn_from_web("anything")
    assert "error" in res
    c.shutdown()


@test("controller web learning enabled with injected fetcher")
def test_controller_enabled():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = _tmp_dir()
    ff = FakeFetcher(wiki={"Photosynthesis": WIKI_PHOTO},
                     searches={"Photosynthesis": ["Photosynthesis"]})
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_agents=False, enable_world_model=False,
        enable_web_learning=True,
        web_data_dir=os.path.join(d, "web"),
        web_fetcher=ff,
    )
    assert c.web_loop is not None
    report = c.learn_from_web("Photosynthesis")
    assert report["facts_added"] > 0, f"report: {report}"
    # Learned knowledge is now retrievable from the controller's own memory
    facts = c.memory.semantic.retrieve("Photosynthesis")
    assert any(f.source.startswith("web:") for f in facts)
    assert c.stats()["web_learning"]["goals_pursued"] >= 1
    c.shutdown()


@test("think() surfaces web-learned knowledge in its answer")
def test_think_uses_learned_knowledge():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = _tmp_dir()
    ff = FakeFetcher(wiki={"Photosynthesis": WIKI_PHOTO},
                     searches={"Photosynthesis": ["Photosynthesis"]})
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_agents=False, enable_world_model=False,
        enable_web_learning=True,
        web_data_dir=os.path.join(d, "web"),
        web_fetcher=ff,
    )
    report = c.learn_from_web("Photosynthesis")
    assert report["facts_added"] > 0
    result = c.think("What is photosynthesis?")
    out = result.primary_output.lower()
    assert "biological process" in out, f"learned fact missing from: {out[:300]}"
    assert "wikipedia" in out          # provenance shown
    assert result.confidence > 0.0
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Fetcher
        test_policy_domain, test_policy_scheme, test_policy_budget,
        test_seen_persist, test_url_hash, test_fake_fetcher,
        # Extractors
        test_text_extract, test_text_title, test_fact_patterns,
        test_fact_pronoun, test_fact_caps,
        # KG writer
        test_kg_writer_persist, test_kg_writer_dedup, test_kg_writer_reload,
        # Belief revision
        test_belief_added, test_belief_reinforced, test_belief_revised,
        test_belief_rejected, test_belief_coexists, test_belief_audit,
        # Curiosity
        test_goal_queue, test_curiosity_hypotheses, test_curiosity_frontier,
        test_curiosity_user,
        # Learning loop
        test_loop_end_to_end, test_loop_batch, test_loop_unknown_topic,
        test_loop_background,
        # Media
        test_media_classify, test_subtitle_parse, test_loop_media_youtube,
        test_loop_media_failure,
        # Controller
        test_controller_disabled, test_controller_enabled,
        test_think_uses_learned_knowledge,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Web Learning Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
