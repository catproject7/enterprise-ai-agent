# enterprise-ai-agent
A production-oriented RAG and Agent platform for enterprise knowledge management.

## Current scope

This repository currently provides:

- Python 3.13 package managed with `uv`
- typed environment-based configuration
- standard-library logging setup
- PDF, Markdown, and TXT document ingestion with unified metadata
- configurable document chunking with overlapping windows and metadata preservation
- pytest and Ruff configuration
- GitHub Actions quality checks

Embeddings, vector storage, RAG, and agents are implemented in later issues and
are intentionally not included yet.

## Development setup

Prerequisites:

- Windows, macOS, or Linux
- `uv`
- Python 3.13, managed automatically by `uv`

Create the environment and install development dependencies:

```powershell
uv sync --dev
```

Create a local environment file when needed:

```powershell
Copy-Item .env.example .env
```

Never commit a real `.env` file.

## Quality checks

Run the test suite:

```powershell
uv run pytest
```

Run lint checks:

```powershell
uv run ruff check .
```

## Configuration

Configuration is loaded from environment variables with the
`ENTERPRISE_AI_AGENT_` prefix. Values can also be provided through a local
`.env` file.

See `.env.example` for the currently supported settings.

## Project structure

```text
src/enterprise_ai_agent/
  __init__.py
  core/
    __init__.py
    config.py
    logging.py
  ingestion/
    __init__.py
    document.py
    exceptions.py
    loaders.py
  chunking/
    __init__.py
    models.py
    splitter.py
tests/
docs/
```

See `docs/architecture.md` for module boundaries and the planned delivery
sequence.
