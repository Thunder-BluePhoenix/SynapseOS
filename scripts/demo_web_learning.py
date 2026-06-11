"""
Demo — SynapseOS Web Learning Loop.

Shows the full cycle: curiosity → fetch → extract → belief revision →
world model perception → knowledge retrievable in think().

Offline (no network, canned fixtures):
    python scripts/demo_web_learning.py --offline

Live (open web, Wikipedia-first):
    python scripts/demo_web_learning.py --topic "Photosynthesis"
    python scripts/demo_web_learning.py --n-goals 3        # curiosity-driven batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapseos.phase5.cognitive_controller import CognitiveController
from synapseos.web import FakeFetcher

OFFLINE_WIKI = {
    "Photosynthesis": {
        "title": "Photosynthesis",
        "extract": ("Photosynthesis is a biological process. It is used by plants "
                    "to convert light into chemical energy. Chlorophyll is a green "
                    "pigment. Photosynthesis was discovered by Jan Ingenhousz."),
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Photosynthesis"}},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="SynapseOS Web Learning demo")
    parser.add_argument("--topic", default="Photosynthesis")
    parser.add_argument("--n-goals", type=int, default=0,
                        help="run N curiosity-driven goals instead of a single topic")
    parser.add_argument("--offline", action="store_true",
                        help="use canned fixtures, no network")
    parser.add_argument("--allow-domain", action="append", default=None,
                        help="restrict to domain(s); default open web")
    args = parser.parse_args()

    fetcher = None
    if args.offline:
        fetcher = FakeFetcher(wiki=OFFLINE_WIKI,
                              searches={args.topic: list(OFFLINE_WIKI)})

    print("=" * 60)
    print("SynapseOS — Web Learning Loop demo")
    print("=" * 60)

    controller = CognitiveController(
        enable_web_learning=True,
        web_allowed_domains=args.allow_domain,
        web_fetcher=fetcher,
        enable_agents=False,
    )
    if controller.web_loop is None:
        print("Web learning failed to initialize — aborting")
        sys.exit(1)

    if args.n_goals > 0:
        print(f"\n→ Curiosity-driven batch: {args.n_goals} goals")
        reports = [r.to_dict() for r in controller.web_loop.run(args.n_goals)]
    else:
        print(f"\n→ Learning topic: {args.topic!r}")
        reports = [controller.learn_from_web(args.topic)]

    for r in reports:
        print(f"\n  topic: {r['topic']}")
        print(f"  pages fetched:   {len(r['urls_fetched'])}  ({r['text_chars']} chars)")
        print(f"  facts extracted: {r['facts_extracted']}")
        print(f"  added: {r['facts_added']}  revised: {r['facts_revised']}  "
              f"rejected: {r['facts_rejected']}  reinforced: {r['facts_reinforced']}")
        print(f"  world-model surprise: {r['surprise']:.3f}")
        if r["errors"]:
            print(f"  errors: {r['errors']}")

    print("\n→ Recent belief revisions:")
    for rev in controller.web_loop.reviser.revisions(limit=5):
        print(f"  [{rev['action']}] {rev['subject']} {rev['predicate']} "
              f"{rev['new_obj']}  ({rev['reason']})")

    topic = args.topic if args.n_goals == 0 else (reports[0]["topic"] if reports else "")
    if topic:
        print(f"\n→ Semantic memory now knows about {topic!r}:")
        for fact in controller.memory.semantic.retrieve(topic, top_k=5):
            print(f"  {fact.subject} —{fact.predicate}→ {fact.obj} "
                  f"(conf {fact.confidence:.2f}, {fact.source})")

        print(f"\n→ think({topic!r}) with the new knowledge:")
        result = controller.think(f"What do you know about {topic}?")
        print(f"  {result.primary_output[:300]}")

    print("\n→ Stats:", controller.web_loop.stats())
    controller.shutdown()
    print("\nDone — knowledge persisted under data/web/ and data/memory/")


if __name__ == "__main__":
    main()
