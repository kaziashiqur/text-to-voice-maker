# Text to Voice Maker

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/kaziashiqur/text-to-voice-maker)](https://github.com/kaziashiqur/text-to-voice-maker/releases)

A free Windows desktop app that turns English and Bangla (বাংলা) scripts into natural voiceovers for YouTube, Facebook and documentaries.

- **Free.** It uses the free tiers of online voice and AI services. No subscription and no credit card are needed.
- **Auto-tag.** An AI reads your script and adds mood tags like `[dramatic]`, `[sad]` and `[pause]`, so the voice acts out the story.
- **Several voice services in one place.** If one service's free limit runs out, the app remakes the script with a similar voice from the next service.
- **English, Bangla, or both mixed** in the same script.

## Voice services

| Service | Best for | Free limit | Get a key |
|---|---|---|---|
| **Gemini** (Google) | Emotional Bangla and English narration | Daily limit; the app switches between Gemini voice models | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **ElevenLabs** | Very natural English | 10,000 characters/month | [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys) |
| **Sarvam** | Native Bengali voices (India) | ₹100 credit on signup | [indus.sarvam.ai](https://indus.sarvam.ai/key-management) |
| **Microsoft Edge** | Clear Bangladeshi and English voices | No key, no limit (plain delivery, no emotion) | — |

You only need one key to start. The Gemini key alone is enough for both voices and auto-tag.

Notes:
- On the ElevenLabs free plan, only ElevenLabs' built-in voices work through the API. Those are English speakers, so they're hidden when the language is set to Bangla. Voice Library voices need a paid plan.
- The Edge voices use the unofficial [edge-tts](https://github.com/rany2/edge-tts) package, which can stop working if Microsoft changes its service.

## Auto-tag services

Auto-tag uses Gemini by default. You can add keys for other free AI services as backups. If one fails or runs out, the next one is tried.

| Service | Get a key |
|---|---|
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) |
| Cerebras | [cloud.cerebras.ai](https://cloud.cerebras.ai/platform) |
| OpenRouter | [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |
| Mistral | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) |

## Install

You need Windows 10 or 11 and [Python 3.12](https://www.python.org/downloads/). In the installer, tick "Add python.exe to PATH".

1. Download the latest version from [Releases](https://github.com/kaziashiqur/text-to-voice-maker/releases/latest) (**Source code (zip)**) and unzip it, or `git clone` this repository.
2. In the project folder, right-click `setup.ps1` → **Run with PowerShell**. This only needs to be done once.
3. Double-click `run.bat` to start the app.
4. Click **⚙ Settings**, paste a free key (see the links above), and click **Test keys**.

## Use

1. Paste or type your script.
2. Click **✨ Auto-tag** to add mood tags, or add them yourself with **Insert tag**.
3. Choose the language and a voice, and click **▶ Preview** to hear it.
4. Click **Generate voice**. The audio is saved as MP3 or WAV in the `output` folder.

### Tags

Mood tags change the voice from that point until the next mood tag:

`[normal]` `[dramatic]` `[curious]` `[serious]` `[concerned]` `[sad]` `[hopeful]` `[calm]` `[warm]` `[happy]` `[excited]` `[urgent]` `[angry]` `[whisper]`

Event tags go between sentences:

`[short pause]` `[pause]` `[long pause]` `[laughs]` `[sighs]` `[breath]`

Example:

```
[dramatic] কল্পনা করুন... [pause]
[concerned] হাজার হাজার শ্রমিক রাস্তায় নেমেছেন।
[curious] কিন্তু কেন?
```

## Privacy

Your API keys are stored only on your computer, in `settings.json` next to the app. That file is never uploaded. Your script text is sent to the voice and AI services you choose.

## Contributing

This is an open-source project and contributions are welcome: bug reports, ideas, testing and code. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE): free to use, change and share.
