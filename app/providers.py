"""All free voice services the app can use, in fallback order.

Each provider turns a piece of text into audio (float32, SR Hz). When a service's free
limit is used up it raises QuotaError and the renderer moves on to the next service.
"""
import asyncio
import base64
import hashlib
import io
import re
import time
from dataclasses import dataclass, field

import numpy as np
import requests

from . import net
import soundfile as sf

from . import config
from .engines import BN_CHAR, SR, resample
from .gemini import GeminiError, QuotaError
from .gemini_tts import VOICES as GEMINI_VOICES
from .gemini_tts import GeminiTTS
from .tags import STYLES

CACHE_DIR = config.CACHE / "tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Voice:
    id: str
    name: str
    gender: str            # "male" | "female"
    lang: str = "multi"    # "multi" (English + Bangla) | "bn" | "en"
    note: str = ""


@dataclass
class Provider:
    id: str
    name: str
    blurb: str
    key_setting: str | None      # settings.json key holding the API key (None = no key needed)
    key_url: str
    free: str                    # how much is free, shown in Settings
    inline: bool                 # True: mood tags are sent inside the text (few, big requests)
    max_chars: int
    voices: list[Voice] = field(default_factory=list)

    def synth(self, text, voice, style, opts, notify) -> np.ndarray:
        raise NotImplementedError


def _decode(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), sr, SR)


def _cached(parts: list[str], make) -> np.ndarray:
    key = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    audio = make()
    np.save(path, audio)
    return audio


def is_bangla(text: str, opts) -> bool:
    return opts.language == "bn" or (opts.language == "auto" and bool(BN_CHAR.search(text)))


# ---------------------------------------------------------------- Gemini
class Gemini(Provider):
    def synth(self, text, voice, style, opts, notify):
        tts = GeminiTTS(opts.keys.get("gemini", ""), opts.gemini_tts_model)
        return tts.synth(text, voice, style, opts.speed, notify=notify, delivery=opts.delivery,
                         inline_tags=True, language=opts.language)


# ---------------------------------------------------------------- ElevenLabs
ELEVEN_MODELS = ["eleven_v3", "eleven_v4"]       # the models that speak Bangla
ELEVEN_DEFAULT_VOICES = [
    Voice("JBFqnCBsd6RMkjVDRZzb", "George", "male", "en"),
    Voice("pNInz6obpgDQGcFmaJgB", "Adam", "male", "en"),
    Voice("onwK4e9ZLuTAKqWW03F9", "Daniel", "male", "en"),
    Voice("21m00Tcm4TlvDq8ikWAM", "Rachel", "female", "en"),
    Voice("EXAVITQu4vr4xnSDxMaL", "Sarah", "female", "en"),
    Voice("XB0fDUnXU5powFXDhCwa", "Charlotte", "female", "en"),
]


def _eleven_text(text: str) -> str:
    """Our tags -> ElevenLabs v3 audio tags (pauses become '...')."""
    text = re.sub(r"\[(short |long )?pause\]\s*", "... ", text)
    text = text.replace("[breath]", "[exhales]")
    return re.sub(r"\[normal\]\s*", "", text)


class ElevenLabs(Provider):
    def synth(self, text, voice, style, opts, notify):
        key = opts.keys.get("elevenlabs", "")
        body_text = _eleven_text(text)
        if style not in ("normal",) and not body_text.lstrip().startswith("["):
            body_text = f"[{style}] {body_text}"
        speed = float(np.clip(opts.speed, 0.7, 1.2))
        lang = "bn" if is_bangla(text, opts) else "en"

        def make():
            last = None
            for model in ELEVEN_MODELS:
                for fmt in ("pcm_24000", "mp3_44100_128"):
                    pass
                    try:
                        r = net.post(
                            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                            params={"output_format": fmt}, timeout=300, notify=notify,
                            label=f"ElevenLabs voice is speaking ({model})…",
                            headers={"xi-api-key": key, "Content-Type": "application/json"},
                            json={"text": body_text, "model_id": model, "language_code": lang,
                                  "voice_settings": {"speed": speed}})
                    except requests.RequestException as e:
                        raise GeminiError(f"ElevenLabs: no internet ({e.__class__.__name__})") from e
                    if r.status_code == 200:
                        if fmt.startswith("pcm"):
                            pcm = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
                            return resample(pcm, 24000, SR)
                        return _decode(r.content)
                    msg = r.text[:300]
                    if "paid_plan_required" in msg or "library voices" in msg:
                        raise GeminiError("ElevenLabs: this is a Voice Library voice, which the free plan "
                                          "can't use. Pick one of ElevenLabs' built-in voices "
                                          "(George, Brian, Sarah, Bill…)")
                    if r.status_code in (402, 429) or "quota" in msg.lower() or "credits" in msg.lower():
                        raise QuotaError("ElevenLabs: this month's free characters are used up")
                    if r.status_code == 401 and "api_key" in msg.lower():
                        raise GeminiError("ElevenLabs: the API key is not valid")
                    last = f"ElevenLabs error {r.status_code}: {msg}"
                    if "output_format" not in msg and "format" not in msg.lower():
                        break                       # not a format problem: try the next model
            raise GeminiError(last or "ElevenLabs failed")

        return _cached(["eleven", voice, lang, f"{speed:.2f}", body_text], make)


def list_eleven_voices(key: str) -> list[Voice]:
    r = net.get("https://api.elevenlabs.io/v2/voices", headers={"xi-api-key": key},
                     params={"page_size": 100}, timeout=30)
    if r.status_code != 200:
        raise GeminiError(f"ElevenLabs: {r.status_code} {r.text[:200]}")
    out = []
    for v in r.json().get("voices", []):
        if v.get("category") == "professional":     # Voice Library voices need a paid plan
            continue
        g = (v.get("labels") or {}).get("gender", "").lower()
        # ElevenLabs' built-in voices are English speakers: Bangla sounds foreign with them.
        lang = "en" if v.get("category") == "premade" else "multi"
        out.append(Voice(v["voice_id"], v.get("name", "?"), "female" if g.startswith("f") else "male", lang))
    return out


# ---------------------------------------------------------------- Sarvam
class Sarvam(Provider):
    def synth(self, text, voice, style, opts, notify):
        key = opts.keys.get("sarvam", "")
        pace = float(np.clip(opts.speed * STYLES.get(style, STYLES["normal"]).speed, 0.5, 2.0))
        lang = "bn-IN" if is_bangla(text, opts) else "en-IN"

        def make():
            for attempt in range(3):
                notify("Sarvam voice is speaking…")
                try:
                    r = net.post("https://api.sarvam.ai/text-to-speech", timeout=300, notify=notify,
                                 label="Sarvam voice is speaking…",
                                      headers={"api-subscription-key": key},
                                      json={"text": text, "language_code": lang, "speaker": voice,
                                            "model": "bulbul:v3", "pace": round(pace, 2),
                                            "speech_sample_rate": 24000})
                except requests.RequestException as e:
                    raise GeminiError(f"Sarvam: no internet ({e.__class__.__name__})") from e
                if r.status_code == 200:
                    return np.concatenate([_decode(base64.b64decode(a)) for a in r.json()["audios"]])
                if r.status_code == 402:
                    raise QuotaError("Sarvam: the free credits are used up")
                if r.status_code == 429 and attempt < 2:
                    net.sleep(10)
                    continue
                if r.status_code in (401, 403):
                    raise GeminiError("Sarvam: the API key is not valid")
                raise GeminiError(f"Sarvam error {r.status_code}: {r.text[:300]}")
            raise QuotaError("Sarvam: too many requests right now")

        return _cached(["sarvam", voice, lang, f"{pace:.2f}", text], make)


# ---------------------------------------------------------------- Microsoft Edge
EDGE_MOOD = {   # style -> (pitch Hz, volume %) ; speed comes from tags.STYLES
    "dramatic": (-3, 0), "curious": (3, 0), "serious": (-2, 0), "concerned": (-2, -5),
    "sad": (-5, -10), "hopeful": (2, 0), "calm": (-1, -5), "warm": (1, 0), "happy": (4, 5),
    "excited": (6, 10), "urgent": (2, 10), "angry": (-2, 15), "whisper": (-3, -40), "normal": (0, 0),
}


class Edge(Provider):
    def synth(self, text, voice, style, opts, notify):
        import edge_tts
        st = STYLES.get(style, STYLES["normal"])
        pitch, vol = EDGE_MOOD.get(style, (0, 0))
        rate = int(round((opts.speed * st.speed - 1) * 100))
        args = dict(rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", volume=f"{vol:+d}%")

        def make():
            notify("Edge voice is speaking…")

            async def run():
                buf = bytearray()
                async for chunk in edge_tts.Communicate(text, voice, **args).stream():
                    if chunk["type"] == "audio":
                        buf.extend(chunk["data"])
                return bytes(buf)
            try:
                data = net.run(lambda: asyncio.run(run()), notify, "Edge voice is speaking…")
            except net.Cancelled:
                raise
            except Exception as e:
                raise GeminiError(f"Edge voice failed ({e.__class__.__name__}: {e})") from e
            if not data:
                raise GeminiError("Edge voice returned no audio")
            return _decode(data)

        return _cached(["edge", voice, str(args), text], make)


# ---------------------------------------------------------------- registry
def _gender(desc: str) -> str:
    return "female" if desc.lower().startswith("female") else "male"


PROVIDERS: list[Provider] = [
    Gemini("gemini", "Gemini", "Best quality, acts out the moods", "gemini_api_key",
           "https://aistudio.google.com/apikey", "about 10 requests/day per model (~10 scripts/day)",
           inline=True, max_chars=2400,
           voices=[Voice(n, n, _gender(d), note=d) for n, d in GEMINI_VOICES]),
    ElevenLabs("elevenlabs", "ElevenLabs", "Very natural, emotional (v3)", "elevenlabs_api_key",
               "https://elevenlabs.io/app/settings/api-keys", "10,000 characters/month (~4 scripts)",
               inline=True, max_chars=2400, voices=list(ELEVEN_DEFAULT_VOICES)),
    Sarvam("sarvam", "Sarvam", "Native Bengali voices (India)", "sarvam_api_key",
           "https://indus.sarvam.ai/key-management", "₹100 free credit on signup (~14 scripts), never expires",
           inline=False, max_chars=2000,
           voices=[Voice("rehan", "Rehan", "male"), Voice("shubh", "Shubh", "male"),
                   Voice("ashutosh", "Ashutosh", "male"), Voice("roopa", "Roopa", "female"),
                   Voice("suhani", "Suhani", "female"), Voice("priya", "Priya", "female")]),
    Edge("edge", "Microsoft Edge", "Clear Bangladeshi voices, no key, no limit", None,
         "", "free, no key needed (unofficial, may change)",
         inline=False, max_chars=3000,
         voices=[Voice("bn-BD-PradeepNeural", "Pradeep (Bangladesh)", "male", "bn"),
                 Voice("bn-BD-NabanitaNeural", "Nabanita (Bangladesh)", "female", "bn"),
                 Voice("bn-IN-BashkarNeural", "Bashkar (India)", "male", "bn"),
                 Voice("bn-IN-TanishaaNeural", "Tanishaa (India)", "female", "bn"),
                 Voice("en-US-AndrewNeural", "Andrew (US English)", "male", "en"),
                 Voice("en-US-BrianNeural", "Brian (US English)", "male", "en"),
                 Voice("en-GB-RyanNeural", "Ryan (UK English)", "male", "en"),
                 Voice("en-US-AvaNeural", "Ava (US English)", "female", "en"),
                 Voice("en-US-EmmaNeural", "Emma (US English)", "female", "en"),
                 Voice("en-GB-SoniaNeural", "Sonia (UK English)", "female", "en")]),
]
BY_ID = {p.id: p for p in PROVIDERS}


def set_eleven_voices(voices: list) -> None:
    """Use the voices from the user's ElevenLabs account (saved in settings)."""
    if voices:
        # older settings saved [id, name, gender] for built-in voices only: those are English
        BY_ID["elevenlabs"].voices = [v if isinstance(v, Voice) else Voice(*v) if len(v) > 3
                                      else Voice(*v, "en") for v in voices]


def lookup(full_id: str) -> tuple[Provider, Voice]:
    pid, _, vid = full_id.partition(":")
    prov = BY_ID.get(pid) or BY_ID["gemini"]
    voice = next((v for v in prov.voices if v.id == vid), None)
    if voice is None:            # e.g. an ElevenLabs voice from another account
        voice = Voice(vid, vid, "male") if vid else prov.voices[0]
    return prov, voice


def has_key(prov: Provider, keys: dict) -> bool:
    return prov.key_setting is None or bool(keys.get(prov.id))


def fallback_chain(full_id: str, keys: dict, bangla: bool) -> list[str]:
    """The chosen voice first, then one similar voice from every other usable service."""
    prov, voice = lookup(full_id)
    chain = [f"{prov.id}:{voice.id}"]
    for p in PROVIDERS:
        if p.id == prov.id or not has_key(p, keys):
            continue
        fits = [v for v in p.voices if v.lang == "multi" or v.lang == ("bn" if bangla else "en")]
        same = [v for v in fits if v.gender == voice.gender] or fits
        if same:
            chain.append(f"{p.id}:{same[0].id}")
    return chain
