# Contributing

Thanks for helping make Text to Voice Maker better! Bug reports, ideas, voice tests in other languages and code are all welcome.

## Reporting a bug or asking for a feature

Open an [issue](https://github.com/kaziashiqur/text-to-voice-maker/issues/new/choose) and pick a template. For bugs, please include:

- what you did and what happened instead
- the voice service and voice you used, and the language setting
- the error message shown in the app (copy the text)

**Never paste your API keys** in an issue, a screenshot or a pull request.

## Working on the code

1. Fork the repository and clone your fork.
2. Run `setup.ps1` once, then start the app with `run.bat` (or `.venv\Scripts\python.exe main.py` to see errors in the console).
3. Make a branch for your change: `git checkout -b my-change`.
4. Test your change in the app with at least one voice service.
5. Commit, push to your fork and open a pull request. Describe what you changed and how you tested it.

### Where things are

| File | What it does |
|---|---|
| `app/ui.py` | The window, settings dialog and buttons |
| `app/synth.py` | Turns tagged text into audio and handles fallback between services |
| `app/providers.py` | The voice services (Gemini, ElevenLabs, Sarvam, Edge) and their voices |
| `app/gemini_tts.py` | Gemini voices: prompts, model rotation and caching |
| `app/llm.py`, `app/gemini.py` | Auto-tag with Gemini and other free AI services |
| `app/tags.py` | Tag parsing and cleanup |
| `app/net.py` | Network calls that can be cancelled |
| `app/config.py` | Settings and folders |

### Guidelines

- Keep the app **free to use**: new services should have a free tier that needs no credit card.
- Keep new dependencies to a minimum and pin their versions in `requirements.txt`.
- Network calls go through `app/net.py` so the Cancel button keeps working.
- Never commit `settings.json`, audio files or API keys.

## Ideas to work on

- Voices and testing for more languages
- A packaged `.exe` so users don't need to install Python
- More free voice services
