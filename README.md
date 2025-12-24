# Loader

Local agentic coding assistant. Runs on your hardware with local LLMs.

## Install

```bash
pip install -e .
```

## Requirements

- Python 3.11+
- Ollama running (`ollama serve`)
- A model pulled (`ollama pull llama3.1:8b`)

## Usage

```bash
# Interactive mode
loader

# Single prompt
loader "Read main.py and explain it"

# Skip confirmation prompts
loader -y "Create a hello.py file"

# Use different model
loader -m qwen2.5:7b
```

**In interactive mode:** type prompts, `clear` to reset, `exit` to quit.

## Tools

- `read` - read files
- `write` - write files
- `edit` - find/replace in files
- `glob` - find files by pattern
- `grep` - search file contents
- `bash` - run shell commands

## License

MIT
