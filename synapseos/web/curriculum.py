"""
Web Learning Loop — Curriculum: structured study plans per subject.

"Knowing physics" isn't a download — it's a learning plan. Each subject maps
to an ordered list of Wikipedia-friendly topics; seeding a subject enqueues
them as curiosity goals (priority 0.7, between user topics and hypotheses),
and the learning loop studies them across sessions. Progress is just the
goal queue's done/pending split.
"""

from __future__ import annotations

from synapseos.web.curiosity import CuriosityEngine

SUBJECTS: dict[str, list[str]] = {
    "mathematics": [
        "Arithmetic", "Algebra", "Geometry", "Trigonometry",
        "Coordinate geometry", "Calculus", "Derivative", "Integral",
        "Probability", "Statistics", "Linear algebra", "Number theory",
        "Mensuration mathematics",
    ],
    "physics": [
        "Classical mechanics", "Newton's laws of motion", "Thermodynamics",
        "Electromagnetism", "Optics", "Wave physics", "Quantum mechanics",
        "Special relativity", "Nuclear physics", "Particle physics",
    ],
    "chemistry": [
        "Atomic structure", "Periodic table", "Chemical bonding",
        "Organic chemistry", "Inorganic chemistry", "Chemical reaction",
        "Acid base chemistry", "Electrochemistry", "Thermochemistry",
    ],
    "biology": [
        "Cell biology", "Genetics", "Evolution", "Human anatomy",
        "Photosynthesis", "Ecology", "Microbiology", "Neuroscience",
        "Molecular biology", "Immune system",
    ],
    "geography": [
        "Physical geography", "Plate tectonics", "Climate",
        "World ocean", "Continent", "Atmosphere of Earth",
        "Human geography", "Cartography", "Biome",
    ],
    "history": [
        "Ancient history", "Roman Empire", "Middle Ages", "Renaissance",
        "Industrial Revolution", "World War I", "World War II",
        "History of India", "Cold War", "History of science",
    ],
    "management": [
        "Management", "Strategic management", "Project management",
        "Operations management", "Marketing", "Organizational behavior",
        "Financial management", "Human resource management", "Leadership",
    ],
    "technology": [
        "Computer science", "Software engineering", "Artificial intelligence",
        "Machine learning", "Computer network", "Database", "Operating system",
        "Cryptography", "Distributed computing",
    ],
    "economics": [
        "Microeconomics", "Macroeconomics", "Supply and demand",
        "Inflation", "Monetary policy", "Game theory", "Behavioral economics",
    ],
    "languages": [
        "Bengali language", "Hindi language", "Spanish language",
        "Japanese language", "English grammar", "Linguistics",
        "Writing system", "Phonetics",
    ],
}


def seed_curriculum(curiosity: CuriosityEngine, subject: str = "all",
                    priority: float = 0.7) -> dict[str, int]:
    """
    Enqueue a subject's study plan (or every subject) as learning goals.
    Returns {subject: goals_added}. Idempotent — duplicates are skipped
    by the goal queue.
    """
    subjects = list(SUBJECTS) if subject == "all" else [subject.lower()]
    added: dict[str, int] = {}
    for name in subjects:
        topics = SUBJECTS.get(name)
        if not topics:
            continue
        count = 0
        for i, topic in enumerate(topics):
            goal = curiosity.add_user_topic(
                topic,
                # earlier topics in a plan are more fundamental → higher priority
                priority=max(0.4, priority - 0.02 * i),
            )
            goal.source = "curriculum"
            goal.reason = f"curriculum: {name}"
            count += 1
        added[name] = count
    return added


def curriculum_progress(curiosity: CuriosityEngine) -> dict[str, dict]:
    """Per-subject done/pending counts from the goal queue."""
    progress: dict[str, dict] = {}
    for goal in curiosity.queue._goals.values():
        if not goal.reason.startswith("curriculum: "):
            continue
        subject = goal.reason.split(": ", 1)[1]
        bucket = progress.setdefault(subject, {"done": 0, "pending": 0, "failed": 0})
        bucket[goal.status] = bucket.get(goal.status, 0) + 1
    return progress
