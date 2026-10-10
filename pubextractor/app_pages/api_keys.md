# API Keys

An API provider runs the model on its servers, so text leaves this machine when the license check allows it.

Put keys in a `.env` file in the folder the app runs from, and keep it out of git:

```text
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

| Provider | Key |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Hugging Face | `HF_TOKEN` |

In Model, choose *API provider*; the card shows whether the key is found. Keys are never shown or logged.
