"""Online auto-tagging with Google Gemini (free API key from https://aistudio.google.com/apikey)."""
import re

import requests

from . import net

from . import tags

BASE = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM_PROMPT = f"""You are an expert voice director for Bangladeshi YouTube/Facebook voiceovers.
You receive a script (Bangla, English or mixed) and return the SAME script with audio tags inserted,
so a text-to-speech voice performs it like a skilled human narrator.

MOOD TAGS - a mood tag sets the voice from that point ON, until the next mood tag:
[dramatic]  hooks, suspense, "imagine..." openings, shocking facts
[curious]   rhetorical or open questions to the audience
[serious]   facts, numbers, economy, politics, official decisions
[concerned] problems, crisis, conflict, suffering, worries
[sad]       loss, death, tragedy
[hopeful]   solutions, positive future, possibilities
[calm]      explanations, background, transitions
[warm]      talking directly and kindly to viewers, call to action (follow, comment, subscribe)
[happy]     good news, joy
[excited]   big announcements, high energy
[urgent]    breaking news, immediate warnings
[angry]     anger, outrage (rare)
[whisper]   secrets (rare)
[normal]    back to a neutral voice

EVENT TAGS - placed between words or sentences:
[short pause] [pause] [long pause] [laughs] [sighs] [breath]

HOW TO DIRECT:
- First understand the whole script: genre, topic and emotional arc. Then tag.
- Put a mood tag at the START of a sentence, only where the mood really changes. Do not repeat
  the same mood tag on consecutive sentences; it carries over automatically.
- A typical narration changes mood every 2-5 sentences. A one-off question can be tagged [curious]
  and then switch back to the previous mood on the next sentence.
- Use [pause] for dramatic effect: after a hook, before a key reveal, after a rhetorical question
  that is answered next. Use [long pause] only at big scene changes. Do not overuse pauses.
- Use [laughs], [sighs], [breath] only when the text clearly calls for it.

STRICT RULES:
- NEVER change, add, remove, translate, fix or reorder any words, letters, numbers or punctuation.
  Only insert tags. Keep all line breaks and blank lines exactly. Keep existing tags.
- Use only the tags listed above, in lowercase, in square brackets.
- Output ONLY the tagged script. No explanations, no markdown, no code fences.

EXAMPLE
Input:
কল্পনা করুন...

হঠাৎ শত শত কারখানা বন্ধ!

কিন্তু কেন?

এই খাতে কাজ করেন প্রায় চল্লিশ লাখ মানুষ।

আপনার মতামত জানান কমেন্টে।

Output:
[dramatic] কল্পনা করুন... [pause]

হঠাৎ শত শত কারখানা বন্ধ!

[curious] কিন্তু কেন? [pause]

[serious] এই খাতে কাজ করেন প্রায় চল্লিশ লাখ মানুষ।

[warm] আপনার মতামত জানান কমেন্টে।"""


class GeminiError(Exception):
    pass


class QuotaError(GeminiError):
    """A service's free limit is used up (the app can switch to another service)."""


def _error(resp) -> GeminiError:
    try:
        msg = resp.json().get("error", {}).get("message", "")
    except ValueError:
        msg = resp.text[:200]
    code = resp.status_code
    if code in (400, 401, 403) and "key" in msg.lower():
        return GeminiError("API key is not valid")
    if code == 429:
        return GeminiError("free usage limit reached, try again later")
    if code == 404:
        return GeminiError("model not found, choose another model in Settings")
    return GeminiError(f"HTTP {code}: {msg}")


# Tried in order after the model chosen in Settings, when a model is retired (404),
# overloaded (503/500), rate-limited (429) or too slow. Lite models come early because
# they stay fast when Google's big models are overloaded.
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
                   "gemini-3.7-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
_busy: dict[str, float] = {}      # model -> time it was found overloaded / too slow
BUSY_FOR = 600


def autotag(text: str, api_key: str, model: str, timeout: int = 75, notify=lambda m: None):
    """Returns (tagged_text, model_used). Tries backup models automatically."""
    import time
    queue = [model] + [m for m in FALLBACK_MODELS if m != model]
    fresh = [m for m in queue if time.time() - _busy.get(m, 0) > BUSY_FOR]
    queue = fresh + [m for m in queue if m not in fresh]      # recently busy models go last
    tried, last_err = [], None
    while queue and len(tried) < 4:
        m = queue.pop(0)
        if m in tried:
            continue
        tried.append(m)
        notify(f"Asking Gemini ({m})…")
        try:
            out = _autotag_once(text, api_key, m, timeout, notify), m
            _busy.pop(m, None)
            return out
        except _Retryable as e:
            last_err = e
            _busy[m] = time.time()
            notify(f"{m} is busy, trying another Gemini model…")
            if e.suggested:            # Google tells us which model to use instead
                queue.insert(0, e.suggested)
    raise GeminiError(f"Gemini models are busy right now (last error: {last_err})")


class _Retryable(GeminiError):
    def __init__(self, msg, suggested=None):
        super().__init__(msg)
        self.suggested = suggested


def _autotag_once(text: str, api_key: str, model: str, timeout: int, notify=None) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.4},
    }
    try:
        r = net.post(f"{BASE}/models/{model}:generateContent", notify=notify,
                     label=f"Asking Gemini ({model})… waiting", timeout=timeout,
                     headers={"x-goog-api-key": api_key}, json=body)
    except requests.Timeout as e:
        raise _Retryable(f"{model} timed out") from e
    except requests.RequestException as e:
        raise GeminiError(f"no internet connection ({e.__class__.__name__})") from e
    if r.status_code in (404, 429, 500, 503):
        m = re.search(r"use models/([\w.\-]+)", r.text)
        raise _Retryable(f"{model}: {_error(r)}", m.group(1) if m else None)
    if r.status_code != 200:
        raise _error(r)
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
        raise GeminiError(f"empty response (reason: {reason})")
    parts = cands[0].get("content", {}).get("parts", [])
    out = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    out = re.sub(r"^```[\w-]*\n|\n?```$", "", out).strip()
    if not out:
        raise GeminiError("empty response")
    return tags.sanitize(out)


def list_models(api_key: str, timeout: int = 15) -> list[str]:
    try:
        r = net.get(f"{BASE}/models", params={"pageSize": 200},
                         headers={"x-goog-api-key": api_key}, timeout=timeout)
    except requests.RequestException as e:
        raise GeminiError(f"no internet connection ({e.__class__.__name__})") from e
    if r.status_code != 200:
        raise _error(r)
    names = []
    for m in r.json().get("models", []):
        name = m.get("name", "").split("/", 1)[-1]
        if "gemini" in name and "generateContent" in m.get("supportedGenerationMethods", []):
            names.append(name)
    return sorted(set(names))
