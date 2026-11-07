"""Online Bangla (and mixed) speech with Gemini TTS, using the same free API key.

Results are cached in .cache/tts so regenerating the same sentence costs no quota.
"""
import base64
import hashlib
import re
import time

import numpy as np
import requests

from . import net

from . import config
from .gemini import BASE, GeminiError, QuotaError, _error

# (voice name, description) -- Gemini prebuilt voices
VOICES = [
    ("Kore", "Female · firm"),
    ("Aoede", "Female · breezy"),
    ("Leda", "Female · youthful"),
    ("Zephyr", "Female · bright"),
    ("Sulafat", "Female · warm"),
    ("Puck", "Male · upbeat"),
    ("Charon", "Male · informative"),
    ("Orus", "Male · firm"),
    ("Fenrir", "Male · excitable"),
    ("Algieba", "Male · smooth"),
]

STYLE_PROMPTS = {
    "normal": "in a natural, clear voice",
    "dramatic": "in a dramatic, suspenseful voice that grabs attention",
    "curious": "in a curious, questioning voice, like asking the listener",
    "serious": "in a serious, confident and informative voice",
    "concerned": "in a concerned, worried voice",
    "sad": "in a sad, emotional voice",
    "hopeful": "in a hopeful, positive voice",
    "calm": "in a calm, steady voice",
    "warm": "in a warm, friendly voice, speaking directly to the viewer",
    "happy": "in a happy, cheerful voice",
    "excited": "in an excited, energetic voice",
    "urgent": "in an urgent, pressing voice",
    "angry": "in an angry, intense voice",
    "whisper": "in a soft whisper",
}

# Overall narration character, applied to every request
DELIVERY = {
    "natural": ("Natural", ""),
    "documentary": ("Documentary narrator", "like a professional documentary narrator, engaging and clear"),
    "news": ("News anchor", "like a professional TV news presenter"),
    "story": ("Storyteller", "like an expressive storyteller"),
    "vlog": ("YouTuber / vlog", "like a friendly YouTuber talking to viewers"),
}

CACHE_DIR = config.CACHE / "tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _speed_words(speed: float) -> str:
    if speed >= 1.35:
        return ", speaking fast"
    if speed >= 1.1:
        return ", speaking a little faster than normal"
    if speed <= 0.7:
        return ", speaking slowly"
    if speed <= 0.9:
        return ", speaking a little slower than normal"
    return ""


def _retry_delay(resp) -> float | None:
    """Seconds Google asks us to wait, or None when it's a daily limit."""
    if "PerDay" in resp.text:
        return None
    try:
        for d in resp.json().get("error", {}).get("details", []):
            if d.get("@type", "").endswith("RetryInfo"):
                return float(str(d.get("retryDelay", "30s")).rstrip("s")) + 1
    except ValueError:
        pass
    return 30.0


_TAG = re.compile(r"\[([a-z ]+)\]\s*")
_EVENT_WORDS = {"laughs": "a short natural laugh", "sighs": "a soft sigh", "breath": "a small breath"}


def _directions(text: str) -> tuple[str, list[str]]:
    """Removes [tags] from the text (Gemini would read them aloud) and turns them into
    director's notes that point at the words where each change starts."""
    clean, marks, pos = "", [], 0
    for m in _TAG.finditer(text):
        clean += text[pos:m.start()]
        pos = m.end()
        name = m.group(1).strip()
        if "pause" in name:
            body = clean.rstrip()
            if body and not body.endswith("..."):
                clean = body + "..." + (clean[len(body):] or " ")
        elif name in STYLE_PROMPTS or name in _EVENT_WORDS:
            marks.append((len(clean), name))
    clean += text[pos:]
    # Quoting the text in notes makes Gemini read the quotes, so only the moods are listed.
    moods = list(dict.fromkeys(n for _, n in marks if n in STYLE_PROMPTS and n != "normal"))
    return clean.strip(), moods


# Each model has its OWN free daily quota, so when one is used up we move to the next.
TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview",
              "gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts"]
_exhausted: dict[str, str] = {}   # model -> date its daily quota ran out


class _ModelUnavailable(GeminiError):
    def __init__(self, model, daily=True):
        super().__init__(model)
        self.daily = daily


class GeminiTTS:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def synth(self, text: str, voice: str, style: str = "normal", speed: float = 1.0,
              notify=lambda m: None, delivery: str = "natural", inline_tags: bool = False,
              language: str = "auto") -> np.ndarray:
        from .engines import BN_CHAR
        how = DELIVERY.get(delivery, DELIVERY["natural"])[1]
        bangla = language == "bn" or (language == "auto" and BN_CHAR.search(text))
        lang = ("natural, fluent Bangladeshi Bangla (read English words in English)" if bangla
                else "natural, fluent English")
        instruction = (f"Read the following text aloud in {lang}"
                       f"{', ' + how if how else ''}, "
                       f"{STYLE_PROMPTS.get(style, STYLE_PROMPTS['normal'])}{_speed_words(speed)}. "
                       f"Pause naturally at '...' and between paragraphs. ")
        if inline_tags:
            text, moods = _directions(text)
            if moods:
                instruction += ("Let your tone follow the meaning of each sentence like a skilled narrator; "
                                f"this passage moves through these moods: {', '.join(moods)}. ")
        instruction += "Read exactly this text, from the first word to the last, nothing else"
        prompt = f"{instruction}:\n{text}"

        key = hashlib.sha1(f"{voice}|{prompt}".encode("utf-8")).hexdigest()
        cached = CACHE_DIR / f"{key}.npy"
        if cached.exists():
            return np.load(cached)

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
        }
        today = time.strftime("%Y-%m-%d")
        models = [self.model] + [m for m in TTS_MODELS if m != self.model]
        for model in models:
            if _exhausted.get(model) == today:
                continue
            try:
                audio = self._request(model, body, voice, notify)
            except _ModelUnavailable as e:
                if e.daily:
                    _exhausted[model] = today
                    notify(f"{model}: free limit used up, switching to another voice model…")
                else:
                    notify(f"{model} is busy, trying another voice model…")
                continue
            np.save(cached, audio)
            return audio
        raise QuotaError("today's free limit is used up on all Gemini voice models (it resets tomorrow)")

    def _request(self, model, body, voice, notify) -> np.ndarray:
        from .engines import SR, resample

        url = f"{BASE}/models/{model}:generateContent"
        for attempt in range(3):
            notify(f"Gemini voice ({voice}, {model}) is speaking…")
            try:
                r = net.post(url, notify=notify, label=f"Gemini voice ({voice}) is speaking…",
                             headers={"x-goog-api-key": self.api_key}, json=body, timeout=300)
            except requests.RequestException as e:
                raise GeminiError(f"no internet connection ({e.__class__.__name__})") from e
            if r.status_code == 200:
                break
            if r.status_code == 429:
                delay = _retry_delay(r)
                if delay is None:                       # daily limit
                    raise _ModelUnavailable(model)
                if attempt < 2 and delay <= 65:         # per-minute limit: wait and retry
                    for left in range(int(delay), 0, -1):
                        notify(f"Gemini free limit: waiting {left} s, then continuing…")
                        net.sleep(1)
                    continue
                raise _ModelUnavailable(model)
            if r.status_code == 404:
                raise _ModelUnavailable(model)
            if r.status_code in (500, 503):
                raise _ModelUnavailable(model, daily=False)
            raise _error(r)

        try:
            part = next(p for p in r.json()["candidates"][0]["content"]["parts"] if "inlineData" in p)
        except (KeyError, IndexError, StopIteration):
            raise GeminiError("Gemini returned no audio") from None
        mime = part["inlineData"].get("mimeType", "")
        m = re.search(r"rate=(\d+)", mime)
        rate = int(m.group(1)) if m else 24000
        pcm = np.frombuffer(base64.b64decode(part["inlineData"]["data"]), dtype="<i2")
        return resample(pcm.astype(np.float32) / 32768.0, rate, SR)
