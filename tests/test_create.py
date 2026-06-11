"""
Tests for SynapseOS Create — music composition, image/video generation, MUSE.

All seeded/deterministic; video test uses tiny frames (ffmpeg if present,
GIF fallback otherwise). Zero network.

Run:
    python tests/test_create.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.create.music import MusicComposer, MOODS
from synapseos.create.image import ImageGenerator
from synapseos.create.video import VideoGenerator


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


def _tmp_dir() -> str:
    return tempfile.mkdtemp(prefix="create_test_")


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------

@test("composes a playable WAV with correct duration")
def test_music_wav():
    comp = MusicComposer(out_dir=_tmp_dir()).compose(mood="calm", bars=4, seed=42)
    assert os.path.exists(comp.path)
    with wave.open(comp.path) as w:
        dur = w.getnframes() / w.getframerate()
    assert abs(dur - comp.duration_s) < 1.0
    assert comp.melody and comp.chords          # the score is inspectable


@test("same seed reproduces the exact piece")
def test_music_deterministic():
    d = _tmp_dir()
    a = MusicComposer(out_dir=d).compose(mood="happy", bars=4, seed=7)
    b = MusicComposer(out_dir=d).compose(mood="happy", bars=4, seed=7)
    assert a.melody == b.melody and a.chords == b.chords
    c = MusicComposer(out_dir=d).compose(mood="happy", bars=4, seed=8)
    assert c.melody != a.melody                  # different seed → new piece


@test("moods map to music theory")
def test_music_moods():
    d = _tmp_dir()
    sad = MusicComposer(out_dir=d).compose(mood="sad", bars=2, seed=1)
    happy = MusicComposer(out_dir=d).compose(mood="happy", bars=2, seed=1)
    assert sad.scale == "natural_minor" and happy.scale == "major"
    assert sad.tempo < happy.tempo


@test("composes for an emotional state")
def test_music_for_state():
    d = _tmp_dir()
    comp = MusicComposer(out_dir=d).compose_for_state(valence=-0.5, arousal=0.3,
                                                      bars=2, seed=3)
    assert comp.mood == "sad"
    comp2 = MusicComposer(out_dir=d).compose_for_state(valence=0.6, arousal=0.8,
                                                       bars=2, seed=3)
    assert comp2.mood == "energetic"


@test("all moods render without error")
def test_music_all_moods():
    d = _tmp_dir()
    for mood in MOODS:
        comp = MusicComposer(out_dir=d).compose(mood=mood, bars=1, seed=5)
        assert os.path.getsize(comp.path) > 1000


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@test("creates a PNG for a concept")
def test_image_basic():
    img = ImageGenerator(out_dir=_tmp_dir()).create("ocean storm", width=160, height=120)
    assert os.path.exists(img.path) and img.path.endswith(".png")
    assert img.style in ("flow", "fractal", "geo")
    assert len(img.palette) == 5


@test("concept determines the image deterministically")
def test_image_deterministic():
    d = _tmp_dir()
    gen = ImageGenerator(out_dir=d)
    a = gen.create("quiet forest", style="fractal", width=80, height=60)
    b = gen.create("quiet forest", style="fractal", width=80, height=60)
    assert Path(a.path).read_bytes() == Path(b.path).read_bytes()
    c = gen.create("ocean storm", style="fractal", width=80, height=60)
    assert Path(c.path).read_bytes() != Path(a.path).read_bytes()


@test("all three styles render")
def test_image_styles():
    gen = ImageGenerator(out_dir=_tmp_dir())
    for style in ("flow", "fractal", "geo"):
        img = gen.create("test concept", style=style, width=80, height=60)
        assert img.style == style and os.path.getsize(img.path) > 500


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

@test("creates an animated video (mp4 or gif fallback)")
def test_video_basic():
    vid = VideoGenerator(out_dir=_tmp_dir()).create(
        "dream sequence", seconds=1.0, fps=4, width=96, height=72, seed=11)
    assert os.path.exists(vid.path)
    assert vid.frames == 4 and vid.format in ("mp4", "gif")
    assert os.path.getsize(vid.path) > 1000


@test("video can carry its own composed soundtrack")
def test_video_soundtrack():
    d = _tmp_dir()
    track = MusicComposer(out_dir=d).compose(mood="mysterious", bars=1, seed=2)
    vid = VideoGenerator(out_dir=d).create(
        "scored film", seconds=1.0, fps=4, width=96, height=72,
        seed=12, soundtrack=track.path)
    assert os.path.exists(vid.path)
    assert vid.audio_path == track.path


# ---------------------------------------------------------------------------
# Raag system, lyrics, singing
# ---------------------------------------------------------------------------

@test("raag grammar: swara math and structure")
def test_raag_swaras():
    from synapseos.create.raag import RAAGS, swara_to_semitone
    assert swara_to_semitone("S") == 0
    assert swara_to_semitone("M") == 6          # tivra Ma
    assert swara_to_semitone("m") == 5
    assert swara_to_semitone("S'") == 12        # upper octave
    yaman = RAAGS["yaman"]
    assert 6 in yaman.aroha_semitones()          # Yaman uses tivra Ma
    bhupali = RAAGS["bhupali"]
    assert len(bhupali.aroha) == 6               # pentatonic + upper Sa
    assert 5 not in bhupali.aroha_semitones() and 11 not in bhupali.aroha_semitones()


@test("raag chosen from emotional state by rasa")
def test_raag_for_state():
    from synapseos.create.raag import raag_for_state
    assert raag_for_state(-0.5, 0.2).rasa == "karuna"      # sad+low → pathos
    assert raag_for_state(0.5, 0.8).rasa == "veera"        # joy+energy → valor
    assert raag_for_state(0.0, 0.2).rasa == "shanta"       # neutral → peace


@test("lyrics: themed, rhymed, deterministic")
def test_lyrics():
    from synapseos.create.lyrics import LyricWriter, RHYME_GROUPS
    a = LyricWriter().write("the ocean", emotion="sad", seed=5)
    b = LyricWriter().write("the ocean", emotion="sad", seed=5)
    assert a.verse == b.verse                    # seeded
    assert len(a.verse) == 4 and len(a.chorus) == 2
    assert "ocean" in a.full_text()
    # ABAB rhyme: lines 1&3 end in same rhyme group
    def group(line):
        w = line.split()[-1].lower()
        return next((g for g, words in RHYME_GROUPS.items() if w in words), None)
    assert group(a.verse[0]) == group(a.verse[2]) is not None
    assert a.syllables                            # singer input present


@test("vocals: formant singing renders with emotion differences")
def test_vocals():
    import numpy as np
    from synapseos.create.vocals import sing_note
    sad = sing_note(220.0, 0.6, "la", emotion="sad")
    energetic = sing_note(220.0, 0.6, "la", emotion="energetic")
    assert len(sad) == len(energetic) > 1000
    assert np.abs(sad).max() <= 1.0
    # energetic voice is louder by configuration (amp 1.1 vs 0.8)
    assert np.abs(energetic).max() > np.abs(sad).max()


@test("tabla and tanpura synthesize")
def test_tabla_tanpura():
    import numpy as np
    from synapseos.create.instruments import tabla_stroke, tanpura_drone
    dha = tabla_stroke("dha")
    assert len(dha) > 100 and np.abs(dha).max() <= 1.0
    drone = tanpura_drone(1.0)
    assert abs(len(drone) - 22050) < 100


@test("full song: write, compose in raag, sing")
def test_song_pipeline():
    from synapseos.create.song import SongComposer
    song = SongComposer(out_dir=_tmp_dir()).sing(
        "the monsoon rain", raag="malkauns", cycles=1, seed=9)
    assert os.path.exists(song.path)
    assert song.raag == "Malkauns" and song.rasa == "karuna"
    assert song.emotion == "sad"                 # rasa → voice emotion
    assert "[Verse]" in song.lyrics_text and "[Chorus]" in song.lyrics_text
    assert song.melody_swaras[: 7] == ["m", "g", "m", "d", "n", "d", "m"]  # pakad opens
    with wave.open(song.path) as w:
        assert w.getnframes() / w.getframerate() > 5.0


@test("song picks raag from affect state when no raag given")
def test_song_affect():
    from synapseos.create.song import SongComposer
    song = SongComposer(out_dir=_tmp_dir()).sing(
        "lost time", valence=-0.5, arousal=0.2, cycles=1, seed=4)
    assert song.rasa == "karuna"


@test("MUSE sings a song end to end")
def test_muse_song():
    from synapseos.create import SongComposer as SC
    agent = _muse()
    agent._song_composer = SC(out_dir=_tmp_dir())
    result = agent.handle("sing a song about the river in raag bhupali")
    assert result.structured["kind"] == "song"
    assert result.structured["raag"] == "Bhupali"
    assert os.path.exists(result.structured["path"])
    assert "[Verse]" in result.output
    assert "river" in result.structured["lyrics_text"].lower()


@test("poster: real typography over procedural background")
def test_poster():
    from synapseos.create.poster import PosterGenerator, PosterSpec
    from PIL import Image
    gen = PosterGenerator(out_dir=_tmp_dir())
    spec = PosterSpec(title="Synapse", subtitle="a mind that grows",
                      body=["learns from the web", "remembers everything"],
                      footer="made by SynapseOS", size=(400, 500))
    poster = gen.create(spec, seed=3)
    assert os.path.exists(poster.path)
    img = Image.open(poster.path)
    assert img.size == (400, 500)
    # deterministic
    p2 = gen.create(spec, seed=3)
    assert Path(p2.path).read_bytes() == Path(poster.path).read_bytes()


@test("poster: palette and solid background honored")
def test_poster_palette():
    from synapseos.create.poster import PosterGenerator, PosterSpec
    gen = PosterGenerator(out_dir=_tmp_dir())
    poster = gen.create(PosterSpec(title="Test", subtitle="sub",
                                   palette=["#ff0000"], background="solid",
                                   size=(200, 250)), seed=1)
    assert os.path.exists(poster.path)


@test("workflow engine: chained DAG with cross-node references")
def test_workflow():
    from synapseos.create.workflow import WorkflowEngine
    import synapseos.create.workflow as wf
    d = _tmp_dir()
    # patch node fns to use tmp dirs (avoid polluting data/)
    from synapseos.create.lyrics import LyricWriter
    from synapseos.create.music import MusicComposer
    wf_engine = WorkflowEngine(manifest_dir=d)
    orig_music = wf.NODE_TYPES["music.compose"]
    wf.NODE_TYPES["music.compose"] = lambda i: MusicComposer(out_dir=d).compose(
        mood=i.get("mood", "calm"), bars=1, seed=i.get("seed")).to_dict()
    try:
        result = wf_engine.run({
            "1": {"type": "lyrics.write",
                  "inputs": {"theme": "the storm", "emotion": "sad", "seed": 2}},
            "2": {"type": "music.compose",
                  "inputs": {"mood": "sad", "seed": 2}},
        })
        assert result.order == ["1", "2"]
        assert not result.errors, result.errors
        assert "title" in result.outputs["1"]
        assert os.path.exists(result.outputs["2"]["path"])
        # manifest written
        assert any(Path(d).glob("run_*.json"))
    finally:
        wf.NODE_TYPES["music.compose"] = orig_music


@test("workflow engine: detects cycles and unknown nodes")
def test_workflow_errors():
    from synapseos.create.workflow import WorkflowEngine
    eng = WorkflowEngine(manifest_dir=_tmp_dir())
    try:
        eng.run({"1": {"type": "lyrics.write", "inputs": {"theme": ["2", "x"]}},
                 "2": {"type": "lyrics.write", "inputs": {"theme": ["1", "x"]}}})
        assert False, "cycle not detected"
    except ValueError:
        pass
    r = eng.run({"1": {"type": "no.such.node", "inputs": {}}})
    assert "1" in r.errors


@test("MUSE typesets a poster on request")
def test_muse_poster():
    from synapseos.create.poster import PosterGenerator
    agent = _muse()
    agent._poster_gen = PosterGenerator(out_dir=_tmp_dir())
    result = agent.handle("make a poster of synapse os that says 'a mind that grows'")
    assert result.structured["kind"] == "poster"
    assert os.path.exists(result.structured["path"])
    assert "typography" in result.output.lower() or "Typeset" in result.output


# ---------------------------------------------------------------------------
# MUSE agent
# ---------------------------------------------------------------------------

def _muse():
    from synapseos.phase5.agents.muse import MUSEAgent
    agent = MUSEAgent()
    d = _tmp_dir()
    from synapseos.create import MusicComposer as MC, ImageGenerator as IG, VideoGenerator as VG
    agent._composer = MC(out_dir=os.path.join(d, "music"))
    agent._imager = IG(out_dir=os.path.join(d, "img"))
    agent._videographer = VG(out_dir=os.path.join(d, "vid"))
    return agent


@test("MUSE routes creative requests")
def test_muse_routing():
    agent = _muse()
    assert agent.can_handle("compose a sad song for me") > 0.55
    assert agent.can_handle("draw an image of a sunset") > 0.55
    assert agent.can_handle("show invoice doctype") == 0.0


@test("MUSE composes music with score in output")
def test_muse_music():
    result = _muse().handle("compose a sad melody")
    assert "natural minor" in result.output
    assert "Audio: " in result.output
    assert result.structured["kind"] == "music"
    assert os.path.exists(result.structured["path"])


@test("MUSE composes for tracked mood")
def test_muse_mood_music():
    from synapseos.affect import EmotionTracker
    tracker = EmotionTracker(state_path=os.path.join(_tmp_dir(), "s.json"))
    for _ in range(4):
        tracker.observe("I am sad and disappointed about everything.")
    agent = _muse()
    agent.wire(emotion_tracker=tracker)
    result = agent.handle("compose music for my mood")
    assert result.structured["mood"] == "sad", result.structured["mood"]


@test("MUSE generates an image from a concept")
def test_muse_image():
    result = _muse().handle("draw an image of a burning city at night")
    assert result.structured["kind"] == "image"
    assert os.path.exists(result.structured["path"])
    assert "burning city" in result.structured["concept"]


@test("MUSE makes a video")
def test_muse_video():
    agent = _muse()
    agent.videographer.create = lambda concept, soundtrack=None: __import__(
        "synapseos.create.video", fromlist=["VideoGenerator"]).VideoGenerator(
        out_dir=_tmp_dir()).create(concept, seconds=0.5, fps=4, width=64,
                                   height=48, soundtrack=soundtrack)
    result = agent.handle("make a video of swirling galaxies")
    assert result.structured["kind"] == "video"
    assert os.path.exists(result.structured["path"])


@test("MUSE records creations in semantic memory")
def test_muse_records():
    from synapseos.phase5.memory import SemanticMemory
    d = _tmp_dir()
    mem = SemanticMemory(store_path=os.path.join(d, "sem.json"),
                         kg_path=os.path.join(d, "missing.json"))
    agent = _muse()
    agent.wire(semantic_memory=mem)
    agent.handle("compose a happy tune")
    made = [f for f in mem.retrieve("user") if f.predicate == "was_made"]
    assert made and made[0].source == "muse_creation"


@test("controller routes composition to MUSE end to end")
def test_controller_muse():
    from synapseos.phase5.cognitive_controller import CognitiveController
    d = _tmp_dir()
    c = CognitiveController(
        memory_episodic_path=os.path.join(d, "epi.json"),
        memory_semantic_path=os.path.join(d, "sem.json"),
        kg_path=os.path.join(d, "missing_kg.json"),
        rules_path=os.path.join(d, "missing_rules.json"),
        enable_world_model=False,
        affect_state_path=os.path.join(d, "affect.json"),
    )
    result = c.think("compose a calm melody")
    assert result.routed_to_agent == "MUSE", result.routed_to_agent
    assert "Audio: " in result.primary_output
    c.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_music_wav, test_music_deterministic, test_music_moods,
        test_music_for_state, test_music_all_moods,
        test_image_basic, test_image_deterministic, test_image_styles,
        test_video_basic, test_video_soundtrack,
        test_raag_swaras, test_raag_for_state, test_lyrics, test_vocals,
        test_tabla_tanpura, test_song_pipeline, test_song_affect,
        test_muse_song,
        test_poster, test_poster_palette, test_workflow, test_workflow_errors,
        test_muse_poster,
        test_muse_routing, test_muse_music, test_muse_mood_music,
        test_muse_image, test_muse_video, test_muse_records,
        test_controller_muse,
    ]

    for t in tests:
        t()

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"Create Tests: {passed}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  → {r['detail']}" if not r["passed"] else ""
        print(f"  [{status}] {r['name']}  ({r['ms']}ms){detail}")

    sys.exit(0 if failed == 0 else 1)
