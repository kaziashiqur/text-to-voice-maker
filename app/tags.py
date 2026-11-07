"""Audio tags: definitions, parsing into render segments, and offline rule-based auto-tagging.

Mood tags  ([serious], [curious], ...) set the voice mood from that point on, until the
next mood tag. [normal] goes back to a neutral voice.
Event tags ([pause], [laughs], ...) insert silence or a sound clip from the sfx folder.
"""
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    speed: float = 1.0   # used by the offline voices (Gemini voices act the mood instead)
    gain: float = 1.0


STYLES = {
    "dramatic": Style(0.90, 1.10),
    "curious": Style(1.00, 1.00),
    "serious": Style(0.95, 1.00),
    "concerned": Style(0.93, 0.95),
    "sad": Style(0.88, 0.80),
    "hopeful": Style(1.00, 1.05),
    "calm": Style(0.93, 0.90),
    "warm": Style(0.97, 1.00),
    "happy": Style(1.05, 1.05),
    "excited": Style(1.12, 1.15),
    "urgent": Style(1.10, 1.15),
    "angry": Style(1.08, 1.25),
    "whisper": Style(0.92, 0.45),
    "normal": Style(),
}

# seconds of silence (or sound clip from sfx/<name>.wav when it exists)
EVENTS = {
    "short pause": 0.25,
    "pause": 0.5,
    "long pause": 1.2,
    "laughs": 0.45,
    "sighs": 0.4,
    "breath": 0.3,
}
SFX_EVENTS = {"laughs", "sighs", "breath"}

ALIASES = {
    "laugh": "laughs", "laughing": "laughs", "sigh": "sighs", "breathes": "breath",
    "whispers": "whisper", "whispering": "whisper", "excitedly": "excited",
    "sadly": "sad", "angrily": "angry", "calmly": "calm", "seriously": "serious",
    "happily": "happy", "neutral": "normal", "question": "curious", "questioning": "curious",
    "worried": "concerned", "hope": "hopeful", "friendly": "warm", "suspense": "dramatic",
}

STYLE_TAGS = list(STYLES)          # includes "normal" (reset), shown last
EVENT_TAGS = list(EVENTS)

SENTENCE_GAP = 0.12
PARAGRAPH_GAP = 0.6

TAG_RE = re.compile(r"\[([^\[\]\n]{1,30})\]")
SENT_SPLIT = re.compile(r"(?<=[.!?।…])\s+|\n")
MARKDOWN_RE = re.compile(r"\*\*|__|^\s{0,3}#{1,6}\s+|^\s*[-*•]\s+", re.M)


def norm_tag(raw: str) -> str:
    t = raw.strip().lower()
    return ALIASES.get(t, t)


def is_known(raw: str) -> bool:
    t = norm_tag(raw)
    return t in STYLES or t in EVENTS


@dataclass
class Segment:
    kind: str                 # speech | pause | sfx
    text: str = ""
    style: str = "normal"
    seconds: float = 0.0
    sfx: str = ""
    gap: str = ""             # for pauses: "sentence" | "paragraph" | "" (from a tag)


def _add_speech(segs, chunk, style):
    if re.search(r"\w", chunk):
        segs.append(Segment("speech", text=chunk.strip(), style=style))


def clean_markdown(text: str) -> str:
    return MARKDOWN_RE.sub("", text)


def parse(text: str) -> list[Segment]:
    segs: list[Segment] = []
    style = "normal"                       # mood carries over until the next mood tag
    paragraphs = [p for p in re.split(r"\n\s*\n", clean_markdown(text).strip()) if p.strip()]
    for pi, para in enumerate(paragraphs):
        if pi:
            segs.append(Segment("pause", seconds=PARAGRAPH_GAP, gap="paragraph"))
        sentences = [s for s in SENT_SPLIT.split(para) if s and s.strip()]
        for si, sent in enumerate(sentences):
            if si:
                segs.append(Segment("pause", seconds=SENTENCE_GAP, gap="sentence"))
            pos = 0
            for m in TAG_RE.finditer(sent):
                _add_speech(segs, sent[pos:m.start()], style)
                t = norm_tag(m.group(1))
                if t in STYLES:
                    style = t
                elif t in SFX_EVENTS:
                    segs.append(Segment("sfx", sfx=t, seconds=EVENTS[t]))
                elif t in EVENTS:
                    segs.append(Segment("pause", seconds=EVENTS[t], sfx=t))
                # unknown tags are skipped (never spoken)
                pos = m.end()
            _add_speech(segs, sent[pos:], style)
    return segs


def strip_tags(text: str) -> str:
    out = TAG_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", out).replace(" \n", "\n").strip()


def sanitize(text: str) -> str:
    """Drop unknown tags and write known ones in their standard form."""
    def fix(m):
        t = norm_tag(m.group(1))
        return f"[{t}]" if t in STYLES or t in EVENTS else ""
    out = TAG_RE.sub(fix, text)
    return re.sub(r"[ \t]{2,}", " ", out)


def _core(s: str) -> str:
    s = strip_tags(clean_markdown(s))
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LNM").lower()


def words_unchanged(original: str, tagged: str) -> bool:
    """True when only tags were added (letters/digits identical)."""
    return _core(original) == _core(tagged)


def transfer_tags(original: str, tagged: str, min_match: float = 0.9) -> str | None:
    """Put the tags an AI chose into the user's ORIGINAL text.

    Small models sometimes change a word or a number while tagging. We line the two texts
    up word by word and copy only the tags, so the user's words are never changed.
    Returns None when the AI changed too much to line up reliably.
    """
    import difflib

    orig_words = [(m.start(), m.group()) for m in re.finditer(r"\S+", original)]
    # tagged text -> list of (tags before this word, word)
    items, pending = [], []
    for tok in re.finditer(r"\[[^\[\]\n]{1,30}\]|\S+?(?=\[|\s|$)", tagged):
        t = tok.group()
        if TAG_RE.fullmatch(t):
            pending.append(t)
        else:
            items.append((pending, t))
            pending = []
    trailing = pending
    a = [_core(w) for _, w in orig_words]
    b = [_core(w) for _, w in items]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    if not a or matched / len(a) < min_match:
        return None
    to_orig = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            to_orig[j + k] = i + k
    inserts: dict[int, list[str]] = {}
    last_i = 0
    for j, (tags_, _) in enumerate(items):
        if j in to_orig:
            last_i = to_orig[j]
        elif tags_:                       # word changed by the AI: use the next matched word
            nxt = next((to_orig[k] for k in range(j + 1, len(items)) if k in to_orig), len(a))
            last_i = nxt
        if tags_:
            inserts.setdefault(last_i, []).extend(tags_)
    out, prev = [], 0
    for idx, (pos, _) in enumerate(orig_words):
        if idx in inserts:
            out.append(original[prev:pos])
            out.append(" ".join(dict.fromkeys(inserts[idx])) + " ")
            prev = pos
    out.append(original[prev:])
    result = "".join(out)
    tail = inserts.get(len(a), []) + trailing
    if tail:
        result = result.rstrip() + " " + " ".join(dict.fromkeys(tail))
    return result
