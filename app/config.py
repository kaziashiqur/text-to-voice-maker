"""Paths and user settings. Everything lives inside the project folder (on D:)."""
import json
import re
import sys
from pathlib import Path

if getattr(sys, "frozen", False):          # running as packaged .exe
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "output"
CACHE = ROOT / ".cache"
SETTINGS_FILE = ROOT / "settings.json"

for d in (OUTPUT, CACHE):
    d.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "gemini_api_key": "",
    "gemini_model": "gemini-3.8-flash",
    "language": "auto",            # auto | en | bn
    "voice": "gemini:Puck",        # "<service>:<voice id>", see providers.py
    "elevenlabs_api_key": "",
    "sarvam_api_key": "",
    "auto_fallback": True,
    "autotag_ai": "gemini",
    "groq_api_key": "",
    "cerebras_api_key": "",
    "openrouter_api_key": "",
    "mistral_api_key": "",
    "delivery": "documentary",
    "gemini_tts_model": "gemini-3.8-flash-tts",
    "speed": 1.0,
    "format": "MP3",
}


def clean_key(text: str) -> str:
    """API key from pasted text: if extra words were pasted around it, keep only the key."""
    text = (text or "").strip()
    if not any(c.isspace() for c in text):
        return text
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-.]{19,}", text)
    tokens = [t.rstrip(".") for t in tokens if any(c.isdigit() for c in t)]
    return max(tokens, key=len) if tokens else text


def load_settings() -> dict:
    data = dict(DEFAULTS)
    loaded = {}
    try:
        loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    for k in list(loaded):
        if k.endswith("_api_key") and isinstance(loaded[k], str):
            loaded[k] = clean_key(loaded[k])
    data.update(loaded)
    if data.get("gemini_model") in RETIRED_MODELS:       # Google retired it for new users
        data["gemini_model"] = DEFAULTS["gemini_model"]
    old = data.pop("bn_voice", None)                     # from the version with offline voices
    if old and old.startswith("gemini:") and "voice" not in loaded:
        data["voice"] = old
    if ":" not in str(data.get("voice", "")):              # older plain Gemini voice name
        data["voice"] = f"gemini:{data.get('voice') or 'Puck'}"
    if data.get("gemini_tts_model") in ("gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"):
        data["gemini_tts_model"] = DEFAULTS["gemini_tts_model"]   # tiny free quota
    for k in ("en_voice", "autotag_mode", "bn_fallback"):
        data.pop(k, None)
    return data


RETIRED_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"}


def save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
