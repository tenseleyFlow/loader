# Recommended Ollama Models for Loader

## Currently Installed

| Model | Size |
|-------|------|
| qwen2.5-coder:14b | 9.0 GB |
| qwen2.5-coder:7b | 4.7 GB |
| qwen2.5:14b | 9.0 GB |
| deepseek-r1:14b | 9.0 GB |
| deepseek-coder-v2:16b | 8.9 GB |
| codellama:13b | 7.4 GB |
| gemma2:9b | 5.4 GB |
| mistral:7b | 4.4 GB |
| llama3.2:3b | 2.0 GB |
| phi3:mini | 2.2 GB |

## Models to Try Next

### Heavy Hitters (best quality, needs more VRAM)

| Model | Size | Why |
|-------|------|-----|
| `qwen2.5-coder:32b` | ~20GB | Best open coding model, rivals GPT-4 on benchmarks |
| `deepseek-r1:32b` | ~20GB | Larger reasoning model, even better multi-step logic |
| `codestral:22b` | ~13GB | Mistral's dedicated coding model, excellent tool use |
| `llama3.3:70b` | ~40GB | Meta's flagship, state-of-the-art instruction following |

### Mid-Size Sweet Spot

| Model | Size | Why |
|-------|------|-----|
| `starcoder2:15b` | ~9GB | BigCode's latest, trained on massive code corpus |
| `granite-code:20b` | ~12GB | IBM's code model, strong at enterprise patterns |
| `yi-coder:9b` | ~5.5GB | 01.AI's coding model, great at code completion |
| `phi4:14b` | ~8GB | Microsoft's latest, punches above its weight |

### Lightweight Speed Demons

| Model | Size | Why |
|-------|------|-----|
| `llama3.3:latest` | ~4.5GB | Latest Llama with improved instruction following |
| `qwen2.5-coder:3b` | ~2GB | Tiny but surprisingly capable for quick tasks |
| `deepseek-r1:7b` | ~4.7GB | Reasoning in a smaller package |
| `codegemma:7b` | ~5GB | Google's code-specific Gemma variant |

## Pull Commands

```bash
# Heavy hitters (if you have the VRAM)
ollama pull qwen2.5-coder:32b
ollama pull deepseek-r1:32b
ollama pull codestral:22b

# Mid-size (recommended next pulls)
ollama pull starcoder2:15b
ollama pull granite-code:20b
ollama pull yi-coder:9b
ollama pull phi4:14b

# Lightweight
ollama pull llama3.3
ollama pull qwen2.5-coder:3b
ollama pull deepseek-r1:7b
ollama pull codegemma:7b
```
