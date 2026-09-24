# enterprise-ai-agent
A production-oriented RAG and Agent platform for enterprise knowledge management.

## Current scope

This repository currently provides:

- Python 3.13 package managed with `uv`
- typed environment-based configuration
- standard-library logging setup
- PDF, Markdown, and TXT document ingestion with unified metadata
- configurable document chunking with overlapping windows and metadata preservation
- FastEmbed-based embeddings and Qdrant-backed vector storage
- semantic vector retrieval from text queries
- a pluggable LLM service abstraction with an OpenAI-compatible provider
- a deterministic RAG context builder with retrieval provenance
- pytest and Ruff configuration
- GitHub Actions quality checks

The retrieval pipeline currently provides:

```text
Document → Ingestion → Chunking → Embedding → Vector Store → Retrieval
```

Retrieval means semantic vector retrieval of chunks. The RAG context builder
turns ordered `SearchResult` objects into bounded context and retains source
metadata. It does not implement prompt construction, LLM orchestration, answer
generation, citation generation, or agents.

The LLM layer currently provides:

```text
Prompt
  ↓
LLMService
  ↓
OpenAI-compatible LLM Provider
```

`LLMService` is an abstraction over synchronous text generation, and
`OpenAICompatibleLLMService` adapts it to an injected OpenAI-compatible client.

This is a service abstraction only. There is still no Context Builder, no Prompt
Builder, no RAG, no Citation handling, and no Agent.

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
  embeddings/
    __init__.py
    fastembed.py
    models.py
    service.py
  vector_store/
    __init__.py
    base.py
    models.py
    qdrant.py
  retrieval/
    __init__.py
    service.py
  llm/
    __init__.py
    openai.py
    service.py
  rag/
    __init__.py
    context.py
tests/
docs/
```

See `docs/architecture.md` for module boundaries and the planned delivery
sequence.
