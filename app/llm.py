"""Auto-tagging with several free AI services. Gemini first, then OpenAI-compatible services.

If a service fails, hits its free limit, or changes the user's words, the next one is tried.
"""
import re
from dataclasses import dataclass, field

import requests

from . import net

from . import gemini, tags
from .gemini import SYSTEM_PROMPT, GeminiError


@dataclass
class TextAI:
    id: str
    name: str
    key_setting: str
    key_url: str
    free: str
    base: str = ""                                    # OpenAI-compatible base URL ("" = Gemini)
    prefer: list[str] = field(default_factory=list)   # model name parts, best first
    only_free: bool = False                           # OpenRouter: only ":free" models


SERVICES = [
    TextAI("gemini", "Gemini", "gemini_api_key", "https://aistudio.google.com/apikey",
           "best for Bangla; same key as the Gemini voices"),
    TextAI("groq", "Groq", "groq_api_key", "https://console.groq.com/keys",
           "no card; very fast; generous daily limit",
           "https://api.groq.com/openai/v1",
           ["gpt-oss-120b", "kimi-k2", "llama-4-maverick", "llama-3.3-70b", "qwen3", "gpt-oss-20b"]),
    TextAI("cerebras", "Cerebras", "cerebras_api_key", "https://cloud.cerebras.ai/platform",
           "no card; about 1 million tokens/day",
           "https://api.cerebras.ai/v1",
           ["gpt-oss-120b", "qwen-3-235b", "llama-4-maverick", "llama-3.3-70b", "llama"]),
    TextAI("openrouter", "OpenRouter", "openrouter_api_key", "https://openrouter.ai/settings/keys",
           "no card; free models, about 50 requests/day",
           "https://openrouter.ai/api/v1",
           ["deepseek-chat", "deepseek-v3", "deepseek", "gpt-oss-120b", "qwen3-235b", "llama-4-maverick",
            "llama-3.3-70b", "gemma-3-27b"], only_free=True),
    TextAI("mistral", "Mistral", "mistral_api_key", "https://console.mistral.ai/api-keys",
           "no card (free 'Experiment' plan, phone number needed)",
           "https://api.mistral.ai/v1",
           ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]),
]
BY_ID = {s.id: s for s in SERVICES}
_model_cache: dict[str, str] = {}


def _pick_model(svc: TextAI, key: str) -> str:
    if svc.id in _model_cache:
        return _model_cache[svc.id]
    r = net.get(f"{svc.base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    if r.status_code in (401, 403):
        raise GeminiError("the API key is not valid")
    if r.status_code != 200:
        raise GeminiError(f"could not list models (HTTP {r.status_code})")
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    if svc.only_free:
        ids = [i for i in ids if i.endswith(":free")]
    ids = [i for i in ids if not re.search(r"guard|whisper|tts|embed|vision|ocr|moderation", i)]
    for part in svc.prefer:
        hit = next((i for i in ids if part in i), None)
        if hit:
            _model_cache[svc.id] = hit
            return hit
    if not ids:
        raise GeminiError("no usable model found")
    _model_cache[svc.id] = ids[0]
    return ids[0]


def _openai_autotag(svc: TextAI, text: str, key: str, notify) -> tuple[str, str]:
    model = _pick_model(svc, key)
    notify(f"Asking {svc.name} ({model})…")
    try:
        r = net.post(f"{svc.base}/chat/completions", timeout=150, notify=notify,
                     label=f"Asking {svc.name} ({model})… waiting",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "temperature": 0.3,
                                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": text}]})
    except requests.Timeout as e:
        raise GeminiError("took too long") from e
    except requests.RequestException as e:
        raise GeminiError(f"no internet connection ({e.__class__.__name__})") from e
    if r.status_code == 429:
        raise GeminiError("free limit reached for now")
    if r.status_code in (401, 403):
        raise GeminiError("the API key is not valid")
    if r.status_code != 200:
        _model_cache.pop(svc.id, None)
        raise GeminiError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        out = r.json()["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, ValueError):
        raise GeminiError("unexpected answer") from None
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    out = re.sub(r"^```[\w-]*\n|\n?```$", "", out).strip()
    if not out:
        raise GeminiError("empty answer")
    return tags.sanitize(out), model


def autotag(text: str, settings: dict, first: str = "gemini", notify=lambda m: None):
    """Returns (tagged_text, "Service (model)", problems). Raises GeminiError if all fail."""
    order = [BY_ID.get(first, SERVICES[0])] + [s for s in SERVICES if s.id != first]
    problems = []
    for svc in order:
        key = settings.get(svc.key_setting, "")
        if not key:
            continue
        try:
            if svc.id == "gemini":
                tagged, model = gemini.autotag(text, key, settings.get("gemini_model") or "gemini-3.8-flash",
                                               notify=notify)
            else:
                tagged, model = _openai_autotag(svc, text, key, notify)
        except GeminiError as e:
            problems.append(f"{svc.name}: {e}")
            continue
        if not tags.words_unchanged(text, tagged):
            fixed = tags.transfer_tags(text, tagged)
            if fixed is None or not tags.words_unchanged(text, fixed):
                problems.append(f"{svc.name}: changed too many words")
                continue
            tagged = fixed       # keep the AI's tags, but the user's exact words
        return tagged, f"{svc.name} ({model})", problems
    if not problems:
        raise GeminiError("no auto-tag service has a key yet. Add a free key in ⚙ Settings → Auto-tag.")
    raise GeminiError("every auto-tag service failed:\n• " + "\n• ".join(problems))


def test_key(svc_id: str, key: str) -> str:
    svc = BY_ID[svc_id]
    if svc.id == "gemini":
        gemini.list_models(key)
        return "works"
    _model_cache.pop(svc.id, None)
    return f"works, will use {_pick_model(svc, key)}"
