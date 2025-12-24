# Loader

A local-first, open-source agentic coding assistant. Think Claude Code, but running on your own hardware with open-weight LLMs.

## Vision

Loader provides a scaffold for running agentic AI coding assistants with:
- **Local LLM backends** - Ollama, llama.cpp, vLLM, or any OpenAI-compatible API
- **Tool system** - File operations, shell commands, code search, and more
- **Agent loop** - Think → Plan → Act → Observe → Repeat
- **Streaming CLI** - Real-time responses in your terminal

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         CLI                              │
│  (Rich terminal UI, streaming, conversation history)    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                      Agent Loop                          │
│  (Think → Select Tool → Execute → Observe → Repeat)     │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    Tools      │   │  LLM Backend  │   │    Context    │
│  - Read       │   │  - Ollama     │   │  - Messages   │
│  - Write      │   │  - llama.cpp  │   │  - Files      │
│  - Bash       │   │  - vLLM       │   │  - Codebase   │
│  - Grep       │   │  - OpenAI API │   │  - Memory     │
│  - Glob       │   └───────────────┘   └───────────────┘
└───────────────┘
```

## Features (Planned)

- [ ] Core agent loop with tool use
- [ ] File operations (read, write, edit)
- [ ] Shell command execution
- [ ] Code search (grep, glob, ripgrep)
- [ ] Ollama backend integration
- [ ] Streaming responses
- [ ] Conversation history
- [ ] Context window management
- [ ] Multi-file awareness
- [ ] Git integration

## Requirements

- Python 3.11+
- Ollama (or other LLM backend)
- A capable GPU (recommended: 8GB+ VRAM for 7B models)

## Quick Start

```bash
# Install
pip install -e .

# Run with Ollama backend
loader --backend ollama --model llama3.1:8b

# Or with a specific task
loader "Help me refactor this function"
```

## Supported Models

Any model that can do tool/function calling:
- Llama 3.1 (8B, 70B) - Excellent tool use
- Qwen 2.5 (7B, 14B, 32B) - Great coding ability
- Mistral/Mixtral - Good general purpose
- DeepSeek Coder - Specialized for code
- CodeLlama - Code-focused

## License

MIT
