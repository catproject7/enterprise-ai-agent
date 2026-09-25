# Architecture

## Current implementation

Issue #1 establishes a minimal Python application foundation:

- `src/enterprise_ai_agent/` is the installable application package.
- `core/config.py` loads typed settings from environment variables.
- `core/logging.py` configures standard-library logging once per process.
- `tests/` verifies only the foundation implemented in this issue.
- `.github/workflows/ci.yml` runs lint and tests for pushes and pull requests.

Issue #2 adds the first ingestion slice:

- `ingestion/document.py` defines the unified `Document` and metadata models.
- `ingestion/exceptions.py` defines ingestion-specific error boundaries.
- `ingestion/loaders.py` loads PDF, Markdown, and TXT files into a `Document`.

Issue #3 adds fixed-size document chunking:

- `chunking/models.py` defines immutable chunk and chunking configuration models.
- `chunking/splitter.py` splits documents by character count and overlap while
  preserving source metadata and offsets.

Issue #4 adds embedding and vector storage:

- `embeddings/` defines a replaceable embedding service and a FastEmbed
  implementation.
- `vector_store/` defines a replaceable vector store and a Qdrant implementation.
- Embedded chunks preserve their complete source metadata and can be retrieved
  by cosine similarity search.

Issue #5 adds a retrieval pipeline:

- `retrieval/service.py` composes an `EmbeddingService` with a `VectorStore`.
- `Retriever` turns query text into an embedding and returns the nearest
  `SearchResult` objects.

Retrieval only performs semantic vector recall. It does not implement RAG
context construction, prompts, LLM generation, Agent behavior, APIs, databases,
or authentication.

Issue #6 adds an LLM service boundary:

- `llm/service.py` defines the `LLMService` abstraction for synchronous text
  generation.
- `llm/openai.py` provides `OpenAICompatibleLLMService`, which sends a prompt
  through an injected OpenAI-compatible client and returns the provider text.

The LLM boundary is:

```text
Prompt
  ↓
LLMService
  ↓
LLM Provider
```

`LLMService` is a standalone generation boundary. It does not depend on
`Retriever`, `SearchResult`, `VectorStore`, or Qdrant, so retrieval and
generation remain independently replaceable.

A future RAG pipeline is expected to compose these boundaries:

```text
Retriever
+
Context Builder
+
Prompt Builder
+
LLMService
→ RAG
```

The RAG layer currently provides the complete orchestration path:

- `rag/context.py` converts ordered `SearchResult` objects into bounded context.
- `Context` retains the selected text and ordered source information, including
  document metadata, chunk index, offsets, retrieval score, and result ID.
- `rag/prompt.py` combines a question, `Context`, and system instructions into a
  deterministic `Prompt` with explicit system, context, and question sections.
- `rag/response.py` defines immutable `Answer`, `Citation`, and `RAGResponse`
  models. Citations preserve result ID, document metadata, chunk index, and
  source offsets without depending on the full retrieval result.
- `rag/pipeline.py` orchestrates `Retriever`, `ContextBuilder`, `PromptBuilder`,
  and `LLMService`, then constructs `Answer` and citations from `Context.sources`.
- `rag/run.py` defines `RAGRun`, a reusable execution trace containing the
  question, actual retrieval results, and final response.

RAGPipeline does not implement sentence-level citation attribution. Its citations
represent the retrieval sources included in the prompt for the answer.

Issue #11 adds an independent offline evaluation layer:

- `evaluation/models.py` defines immutable datasets, cases, and metric results.
- `evaluation/metrics.py` calculates Recall@K, Precision@K, normalized
  exact-match Answer correctness, and Citation Coverage.
- `evaluation/evaluator.py` aggregates metrics for already-produced retrieval
  results and RAG responses without calling production components.

Issue #12 adds batch evaluation orchestration:

- `evaluation/runner.py` executes an `EvaluationDataset` through
  `RAGPipeline.run_with_trace()` and invokes `Evaluator` for each case.
- `EvaluationReport` contains ordered per-case results and aggregate metrics.
- The first version is offline, deterministic, dependency-injected, and fail-fast.

Issue #13 adds a portable dataset format and loader:

- `evaluation/models.py` adds optional dataset metadata with name and version.
- `evaluation/dataset.py` loads UTF-8 JSON into a validated `EvaluationDataset`.
- Ground-truth source, chunk index, and offsets are validated without normalization.

Evaluation does not modify production RAG behavior, repeat retrieval, use an LLM
judge, semantic similarity, an external evaluation framework, a CLI, or a dashboard.

Issue #14 adds a minimal tool boundary:

- `tools/base.py` defines the synchronous `Tool[InputT, OutputT]` abstraction
  with a stable name, description, and `run()` method.
- `tools/rag.py` adapts an injected `RAGPipeline` through `RAGTool`.
- `RAGTool` passes queries and responses through unchanged and does not repeat
  retrieval, prompting, generation, citation, or validation logic.

The tool boundary is:

```text
Future Agent
  ↓
Tool
  ↓
RAGPipeline
  ↓
RAGResponse
```

Agent orchestration, tool registries, tool calling, and multiple tools are not
implemented.

## Module boundaries

New modules should be introduced only when the corresponding issue needs them.
The expected responsibilities are:

| Area | Responsibility |
| --- | --- |
| `core` | Cross-cutting configuration, logging, and later shared infrastructure |
| Ingestion | Load and parse PDF, Markdown, and TXT documents with metadata |
| Chunking | Split parsed documents by size and overlap while preserving metadata |
| Embedding and vector storage | Embed chunks, index them in Qdrant, and perform similarity search |
| Retrieval | Embed query text and retrieve the nearest stored chunks |
| LLM | Generate a synchronous text response for a prompt through a provider |
| RAG context | Build bounded context while retaining retrieval provenance |
| RAG prompt | Build a deterministic prompt from a question and retrieved context |
| RAG response | Model final answers and their ordered supporting citations |
| RAG pipeline | Orchestrate retrieval, prompting, generation, and citations |
| RAG | Retrieve context, build prompts, generate answers, and return citations |
| Tools | Provide stable synchronous capabilities for a future Agent |
| API | Expose application capabilities through FastAPI |
| Persistence | Store users, documents, and conversations in PostgreSQL |
| Authentication | Authenticate users and enforce permissions |
| Agents | Orchestrate tools and tool calling |
| Evaluation | Measure retrieval and answer quality against benchmark datasets |

## Planned delivery sequence

The initial delivery sequence keeps each issue narrowly scoped:

1. Issue #1: project architecture and engineering foundation.
2. Issue #2: document ingestion for PDF, Markdown, and TXT files, including
   parsing and metadata. This issue is implemented and does not perform
   chunking.
3. Issue #3: document chunking with chunk size, overlap, metadata preservation,
   and batch processing. This issue is implemented and does not perform
   embedding or retrieval.
4. Issue #4: embedding abstraction and implementation, Qdrant storage,
   document indexing, and vector similarity search. This issue is implemented
   and does not implement the complete RAG pipeline or agents.
5. Issue #5: semantic retrieval from query text through embedding and vector
   search. This issue is implemented and does not perform context construction,
   prompt handling, LLM generation, source citations, or agent orchestration.
6. Issue #6: an `LLMService` abstraction and an OpenAI-compatible adapter for
   synchronous text generation. This issue is implemented and does not perform
   context construction, prompt building, source citations, RAG, or agent
   orchestration.
7. Next RAG slices: deterministic context and prompt construction, stable answer
   and citation models, and pipeline orchestration. This path is implemented.
   Citations represent the prompt context sources, not sentence-level attribution.
8. Issue #11: an offline evaluation foundation for retrieval, answers, and
   citations. This issue is implemented without entering the production RAG
   call chain.
9. Issue #12: a batch evaluation runner using RAG execution traces and
   aggregated reports. This issue is implemented with fail-fast offline execution.
10. Issue #13: JSON evaluation dataset metadata and loading. This issue is
    implemented without running the RAG pipeline or adding a CLI.
11. Issue #14: a minimal Tool abstraction and `RAGTool` adapter for the future
    Agent boundary. This issue does not implement Agent orchestration.
12. Later issues: citation attribution, observability, agents, APIs, persistence,
    authentication, and production hardening.

Hybrid retrieval, BM25, reranking, agents, FastAPI APIs, PostgreSQL,
authentication, and permissions belong to later issues and must not be added
early.
