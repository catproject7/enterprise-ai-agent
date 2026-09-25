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
- a deterministic RAG prompt builder with structured sections
- immutable RAG answer and citation response models
- an end-to-end RAG pipeline orchestration
- a minimal Agent foundation for deterministic single-tool execution
- finite LLM tool calling with one supported tool-call round
- a FastAPI application foundation for health and Agent execution endpoints
- an in-memory Conversation persistence foundation
- standard-library observability with request IDs and structured events
- a runtime composition root for assembling the complete Agent stack
- a minimal Tool boundary with a RAGPipeline adapter for future agents
- offline RAG evaluation metrics, JSON datasets, and batch evaluation runner
- pytest and Ruff configuration
- GitHub Actions quality checks

The retrieval pipeline currently provides:

```text
Document → Ingestion → Chunking → Embedding → Vector Store → Retrieval
```

Retrieval means semantic vector retrieval of chunks. The RAG context builder
turns ordered `SearchResult` objects into bounded context and retains source
metadata. PromptBuilder then combines a question, Context, and system
instructions into a deterministic Prompt. RAGPipeline composes retrieval, context,
prompt construction, and LLM generation into a RAGResponse. Citations represent
the retrieval sources included in the prompt; sentence-level attribution is not
implemented. Evaluation is an offline layer outside RAGPipeline and provides
Recall@K, Precision@K, normalized exact-match Answer correctness, and Citation
Coverage. EvaluationRunner executes an EvaluationDataset through RAGPipeline,
evaluates each trace, and returns an aggregated EvaluationReport. Evaluation
datasets can be loaded from UTF-8 JSON files with optional version metadata.
It does not use an LLM judge, parallel execution, or an external evaluation
framework.

The Agent and Tool Calling layers currently provide:

```text
User
  ↓
LLMAgent
  ↓
ToolCallingLLM
  ↓
ToolRegistry
  ↓
Tool
  ↓
RAGTool
  ↓
RAGPipeline
```

`Agent` defines the execution boundary. `ToolAgent` deterministically runs one
injected Tool, while `LLMAgent` supports at most one LLM-requested Tool Call
followed by one final answer. `RAGTool` adapts the existing `RAGPipeline`
without duplicating RAG behavior. Parallel Tool Calls, repeated Agent loops,
Memory, Planning, and streaming are not implemented.

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

`LLMService` remains the text-generation boundary. Tool Calling is provided by
the separate provider-neutral `ToolCallingLLM` abstraction.

The HTTP API currently provides:

```text
HTTP Request
  ↓
Agent[str]
  ↓
AgentResult[str]
  ↓
HTTP Response
```

The API layer depends only on the `Agent[str]` abstraction. It does not import
LLM providers, RAG implementation, vector storage, or tools.

The Conversation layer currently provides:

```text
FastAPI
  ↓
ConversationService
  ├── ConversationStore
  └── Agent[str]
```

`ConversationService` assembles deterministic history input and keeps the Agent
stateless. `InMemoryConversationStore` stores conversations for the current
process only; data is lost when the process restarts. A persistent backend can
replace the store implementation later without changing the Agent contract.

The Observability layer currently provides:

- request IDs and request timing at the FastAPI boundary
- standard-library structured log events
- transparent Agent, LLM, and Tool wrapper instrumentation
- deterministic EvaluationReport JSON serialization

No OpenTelemetry, Prometheus, Grafana, or Jaeger stack is included.

## Runtime Composition

`enterprise_ai_agent.runtime` assembles the existing components into a
RAG-capable Agent:

```text
Settings
  ↓
EmbeddingService + VectorStore
  ↓
Retriever → RAGPipeline → RAGTool
  ↓
ToolRegistry + ToolCallingLLM → LLMAgent
  ↓
FastAPI
```

Create a runtime programmatically:

```python
from enterprise_ai_agent.runtime import create_runtime_app

app = create_runtime_app()
```

Start the runtime with Uvicorn:

```powershell
$env:ENTERPRISE_AI_AGENT_LLM_API_KEY = "your-api-key"
uv run uvicorn enterprise_ai_agent.runtime.composition:create_runtime_app --factory
```

Runtime settings use these environment variables:

- `ENTERPRISE_AI_AGENT_LLM_MODEL`
- `ENTERPRISE_AI_AGENT_LLM_API_KEY`
- `ENTERPRISE_AI_AGENT_LLM_BASE_URL`
- `ENTERPRISE_AI_AGENT_EMBEDDING_MODEL`
- `ENTERPRISE_AI_AGENT_EMBEDDING_DIMENSION`
- `ENTERPRISE_AI_AGENT_EMBEDDING_BATCH_SIZE`
- `ENTERPRISE_AI_AGENT_EMBEDDING_CACHE_DIR`
- `ENTERPRISE_AI_AGENT_QDRANT_URL`
- `ENTERPRISE_AI_AGENT_QDRANT_API_KEY`
- `ENTERPRISE_AI_AGENT_QDRANT_COLLECTION_NAME`

The default `enterprise_ai_agent.api.app:app` remains health-only and does
not create an Agent unless one is explicitly injected.

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

## FastAPI application

Start the application:

```powershell
uv run uvicorn enterprise_ai_agent.api.app:app --reload
```

Available endpoints:

- `GET /health`
- `POST /agent/run`
- `POST /conversations`
- `GET /conversations/{conversation_id}`
- `POST /conversations/{conversation_id}/messages`

`POST /agent/run` accepts `{"input": "..."}`. The default application has no
real Agent configured, so it returns `503 Service Unavailable` for this endpoint
until an `Agent[str]` is injected through `create_app(agent)`.

## Configuration

Configuration is loaded from environment variables with the
`ENTERPRISE_AI_AGENT_` prefix. Values can also be provided through a local
`.env` file.

See `.env.example` for the currently supported settings.

## Project structure

```text
src/enterprise_ai_agent/
  __init__.py
  api/
    __init__.py
    app.py
    dependencies.py
    models.py
    routes.py
  conversation/
    __init__.py
    exceptions.py
    models.py
    service.py
    store.py
  observability/
    __init__.py
    context.py
    instrumentation.py
    logging.py
    middleware.py
  runtime/
    __init__.py
    composition.py
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
    openai_tool_calling.py
    service.py
    tool_calling.py
  rag/
    __init__.py
    context.py
    pipeline.py
    prompt.py
    response.py
  agent/
    __init__.py
    base.py
    exceptions.py
    llm.py
    models.py
    tool.py
  evaluation/
    __init__.py
    dataset.py
    evaluator.py
    export.py
    metrics.py
    models.py
    runner.py
  tools/
    __init__.py
    base.py
    exceptions.py
    rag.py
    registry.py
tests/
docs/
```

See `docs/architecture.md` for module boundaries and the planned delivery
sequence.
