# Local Model

A model on this machine reads the text without sending it anywhere, through [Ollama](https://ollama.com).

## Set up

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Pull a model: `ollama pull gemma3:4b`
3. In Model, choose *Ollama (this machine)* and the model.

## Sizes

| Model | Download | Reads images |
|---|---|---|
| `gemma3:1b` | 0.8 GB | no |
| `gemma3:4b` | 3.3 GB | yes |
| `gemma3:12b` | 8.1 GB | yes |
| `llama3.1:8b` | 4.9 GB | no |

The Model step shows the free space on the model drive. Image tables need a model that reads images.

Optional: `pip install "pubextractor[semantic]"` finds passages by meaning (about 1-2 GB).
