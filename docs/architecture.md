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
| RAG | Retrieve context, build prompts, generate answers, and return citations |
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
6. Later issues: baseline RAG with context construction, prompt handling, LLM
   generation, source citations, and an end-to-end flow.

Hybrid retrieval, BM25, reranking, agents, FastAPI APIs, PostgreSQL,
authentication, and permissions belong to later issues and must not be added
early.
