"""
Web Learning Loop — Extraction.

Two stages:
  1. TextExtractor — HTML → clean prose (stdlib html.parser; trafilatura
     used automatically when installed).
  2. FactExtractor — prose → (subject, predicate, object) triples via
     sentence-level regex patterns. Same philosophy as the Phase 1 rules
     extractor: deterministic, auditable, no LLM in the loop. Every fact
     keeps its source sentence as provenance.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser

try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "nav", "footer", "aside", "header", "noscript",
              "form", "button", "iframe", "svg"}
_KEEP_TAGS = {"p", "li", "h1", "h2", "h3", "td", "blockquote", "figcaption"}
_MIN_BLOCK_CHARS = 30   # boilerplate heuristic — short fragments are menus/labels


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._capture_depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _KEEP_TAGS and self._skip_depth == 0:
            self._capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _KEEP_TAGS and self._capture_depth > 0:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                block = re.sub(r"\s+", " ", " ".join(self._buf)).strip()
                self._buf = []
                if len(block) >= _MIN_BLOCK_CHARS:
                    self.blocks.append(block)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0 and self._capture_depth > 0:
            self._buf.append(data)


class TextExtractor:
    """HTML → clean text. Drops scripts, navigation, and short boilerplate."""

    def extract(self, html_text: str) -> str:
        if _HAS_TRAFILATURA:
            try:
                out = trafilatura.extract(html_text)
                if out:
                    return out.strip()
            except Exception:
                pass
        parser = _TextHTMLParser()
        try:
            parser.feed(html_text)
        except Exception:
            pass
        if parser.blocks:
            return "\n".join(parser.blocks)
        # Not HTML at all? Treat as plain text.
        if "<" not in html_text[:200]:
            return html_text.strip()
        return ""

    def extract_title(self, html_text: str) -> str:
        parser = _TextHTMLParser()
        try:
            parser.feed(html_text)
        except Exception:
            pass
        return re.sub(r"\s+", " ", html.unescape(parser.title)).strip()


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    obj: str
    confidence: float    # pattern prior, refined later by belief revision
    sentence: str        # provenance — the sentence this came from
    pattern: str         # which rule fired

    def to_dict(self) -> dict:
        return asdict(self)


# Noun-phrase fragment: starts with capital or digit, modest length.
_NP = r"([A-Z0-9][\w''&.,()\- ]{1,60}?)"
# Object phrase: looser — lowercase allowed (e.g. "a dwarf planet").
_OBJ = r"([\w''&.\- ][\w''&.,()\- ]{1,70}?)"

# (predicate, compiled regex, base confidence)
_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("created_by", re.compile(
        rf"^{_NP}\s+(?:was|were)\s+(?:created|invented|founded|developed|designed|written|built)\s+by\s+{_OBJ}$"), 0.65),
    ("born_in", re.compile(rf"^{_NP}\s+was\s+born\s+(?:in|on)\s+{_OBJ}$"), 0.6),
    ("died_in", re.compile(rf"^{_NP}\s+died\s+(?:in|on)\s+{_OBJ}$"), 0.6),
    ("part_of", re.compile(rf"^{_NP}\s+(?:is|are)\s+(?:a\s+)?part\s+of\s+{_OBJ}$"), 0.6),
    ("located_in", re.compile(
        rf"^{_NP}\s+(?:is|are|was|were)\s+(?:located|situated)?\s*in\s+{_OBJ}$"), 0.6),
    ("used_for", re.compile(rf"^{_NP}\s+(?:is|are)\s+used\s+(?:for|to|in)\s+{_OBJ}$"), 0.5),
    ("causes", re.compile(
        rf"^{_NP}\s+(?:causes?|leads?\s+to|results?\s+in)\s+{_OBJ}$"), 0.5),
    ("is_a", re.compile(
        rf"^{_NP}\s+(?:is|are|was|were)\s+(?:a|an|the)\s+{_OBJ}$"), 0.6),
    ("has_property", re.compile(
        rf"^{_NP}\s+(?:has|have|contains?|consists?\s+of)\s+{_OBJ}$"), 0.45),
]

_PRONOUNS = {"it", "they", "he", "she", "this", "these", "the process"}
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class FactExtractor:
    """Pattern-based triple extraction. Deterministic and auditable."""

    def __init__(self, max_facts_per_page: int = 40):
        self.max_facts_per_page = max_facts_per_page

    def extract(self, text: str, topic_hint: str | None = None) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str, str]] = set()

        for sentence in self._split_sentences(text):
            if len(facts) >= self.max_facts_per_page:
                break
            sent = sentence.strip().rstrip(".!?")
            if not (10 <= len(sent) <= 300):
                continue
            sent = self._resolve_pronoun(sent, topic_hint)
            for predicate, pattern, conf in _PATTERNS:
                m = pattern.match(sent)
                if not m:
                    continue
                subject = self._normalize_entity(m.group(1))
                obj = self._normalize_entity(m.group(2))
                if not subject or not obj or subject.lower() == obj.lower():
                    break
                key = (subject.lower(), predicate, obj.lower())
                if key not in seen:
                    seen.add(key)
                    facts.append(ExtractedFact(
                        subject=subject, predicate=predicate, obj=obj,
                        confidence=conf, sentence=sentence.strip(), pattern=predicate,
                    ))
                break   # first matching pattern wins per sentence
        return facts

    def _split_sentences(self, text: str) -> list[str]:
        out: list[str] = []
        for line in text.split("\n"):
            out.extend(_SENTENCE_SPLIT.split(line))
        return out

    def _resolve_pronoun(self, sentence: str, topic_hint: str | None) -> str:
        if not topic_hint:
            return sentence
        first_word = sentence.split(" ", 1)[0].lower()
        if first_word in _PRONOUNS:
            rest = sentence.split(" ", 1)[1] if " " in sentence else ""
            verb = rest.split(" ", 1)[0].lower() if rest else ""
            # "It is a ..." → "<Topic> is a ..."; fix plural agreement
            if verb == "are":
                rest = "is" + rest[3:]
            return f"{topic_hint.strip().title()} {rest}"
        return sentence

    def _normalize_entity(self, s: str) -> str:
        s = _ARTICLE_RE.sub("", s.strip())
        s = re.sub(r"\s+", " ", s).strip(" ,;:")
        if not (2 <= len(s) <= 70):
            return ""
        # Reject fragments that are mostly stopwords/punctuation
        if not re.search(r"[A-Za-z]{2,}", s):
            return ""
        return s
