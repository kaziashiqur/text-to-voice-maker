"""Turns tagged text into one audio array, using the chosen voice service.

If that service's free limit is used up (or it fails), the whole script is made again with
a similar voice from the next free service, so the voice never changes halfway through.
"""
from dataclasses import dataclass, field

import numpy as np

from . import net, providers
from .engines import SR
from .gemini import GeminiError
from .tags import parse, strip_tags


@dataclass
class RenderOptions:
    language: str = "auto"              # auto | en | bn
    voice: str = "gemini:Puck"          # "<provider>:<voice id>"
    speed: float = 1.0
    keys: dict = field(default_factory=dict)   # provider id -> API key
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    delivery: str = "documentary"       # see gemini_tts.DELIVERY
    auto_fallback: bool = True


Cancelled = net.Cancelled


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), np.float32)


def _inline_chunks(segs, max_chars):
    """Few big pieces; mood changes / pauses written inline as [tags]."""
    chunks, pending, lead = [[]], [], 0.0
    for seg in segs:
        if seg.kind != "speech":
            if chunks[-1]:
                pending.append(seg)
            else:
                lead = max(lead, seg.seconds)
            continue
        cur = chunks[-1]
        if cur:
            size = sum(len(p) + len(s.text) for p, s in cur)
            para = any(p.gap == "paragraph" for p in pending)
            if size > max_chars or (size > max_chars * 0.67 and para):
                chunks.append([])
                cur = chunks[-1]
        prefix = ""
        if cur:
            tags_ = [f"[{p.sfx}]" for p in pending if p.kind == "sfx" or p.gap == ""]
            prefix = "\n\n" if any(p.gap == "paragraph" for p in pending) else " "
            prefix += "".join(t + " " for t in tags_)
            if seg.style != cur[-1][1].style:
                prefix += f"[{seg.style}] "
        cur.append((prefix, seg))
        pending.clear()
    tail = max((p.seconds for p in pending), default=0.0)
    return [c for c in chunks if c], lead, tail


def _style_pieces(segs, max_chars):
    """One piece per run of sentences with the same mood; pauses become real silence."""
    items, cur = [], None          # items: ("speech", text, style, n) | ("pause", seconds)
    gap = ""
    for seg in segs:
        if seg.kind == "speech":
            if cur and cur[2] == seg.style and len(cur[1]) + len(seg.text) < max_chars:
                cur[1] += ("\n" if gap == "paragraph" else " ") + seg.text
                cur[3] += 1
            else:
                cur = ["speech", seg.text, seg.style, 1]
                items.append(cur)
            gap = ""
        elif seg.kind == "pause" and seg.gap in ("sentence", "paragraph"):
            gap = "paragraph" if seg.gap == "paragraph" or gap == "paragraph" else "sentence"
            if gap == "paragraph" and cur is not None:
                cur = None if len(cur[1]) > max_chars * 0.5 else cur
                if cur is None:
                    items.append(("pause", 0.5))
        else:                      # [pause] tags and [laughs] etc.
            items.append(("pause", seg.seconds if seg.kind == "pause" else 0.3))
            cur, gap = None, ""
    return items


def _render_with(prov, voice, segs, n_speech, opts, progress, cancel):
    done = 0

    def notify_at(msg):
        progress(int(done * 100 / n_speech), msg)

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    out = []
    if prov.inline:
        chunks, lead, tail = _inline_chunks(segs, prov.max_chars)

        def say(items, depth=0):
            text_ = "".join(p + s.text for p, s in items).strip()
            audio = prov.synth(text_, voice.id, items[0][1].style, opts, notify_at)
            expected = len(strip_tags(text_)) / 15.0
            if len(audio) / SR < 0.45 * expected and len(items) > 1 and depth < 3:
                mid = len(items) // 2       # the service stopped early: split and retry
                return np.concatenate([say(items[:mid], depth + 1), _silence(0.3),
                                       say(items[mid:], depth + 1)])
            return audio

        if lead:
            out.append(_silence(lead))
        for i, c in enumerate(chunks):
            check()
            notify_at(f"Generating… part {i + 1}/{len(chunks)}")
            if i:
                out.append(_silence(0.6))
            out.append(say(c))
            done += len(c)
        if tail:
            out.append(_silence(tail))
    else:
        items = _style_pieces(segs, prov.max_chars)
        n_parts = sum(it[0] == "speech" for it in items)
        k = 0
        for it in items:
            check()
            if it[0] == "pause":
                out.append(_silence(it[1]))
                continue
            k += 1
            notify_at(f"Generating… part {k}/{n_parts}")
            out.append(prov.synth(it[1], voice.id, it[2], opts, notify_at))
            out.append(_silence(0.15))
            done += it[3]
    return np.concatenate(out).astype(np.float32) if out else _silence(0.1)


def render(text: str, opts: RenderOptions, progress=lambda pct, msg: None,
           cancel=None, notes: list | None = None) -> np.ndarray:
    notes = notes if notes is not None else []
    segs = parse(text)
    n_speech = sum(s.kind == "speech" for s in segs)
    if not n_speech:
        raise ValueError("Nothing to read: the text is empty or contains only tags.")

    bangla = providers.is_bangla(text, opts)
    chain = (providers.fallback_chain(opts.voice, opts.keys, bangla) if opts.auto_fallback
             else [opts.voice])
    problems = []
    for full_id in chain:
        prov, voice = providers.lookup(full_id)
        if not providers.has_key(prov, opts.keys):
            problems.append(f"{prov.name}: no API key (add it in ⚙ Settings)")
            continue
        try:
            audio = _render_with(prov, voice, segs, n_speech, opts, progress, cancel)
        except Cancelled:
            raise
        except GeminiError as e:
            problems.append(f"{prov.name}: {e}")
            continue
        if problems:
            notes.append("; ".join(problems) + f". Used {prov.name} · {voice.name} instead.")
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 0.98:
            audio *= 0.98 / peak
        return audio
    raise GeminiError("No voice service could make this audio:\n• " + "\n• ".join(problems))
