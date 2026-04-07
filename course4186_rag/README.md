# Course 4186 RAG Pipeline

This folder contains the standalone knowledge pipeline that supports the Course 4186 student learning system. In the bundled deployment, the active portal uses the packaged lecture PDFs under `course4186_materials` together with the packaged full-course artifacts under `artifacts_full_course`.

The original project code under `D:\digital_human` is not modified. All new code and artifacts stay in this folder.

## Why a new pipeline was created

The existing project RAG implementation is not a safe drop-in choice for Course 4186:

1. `D:\digital_human\rag.py` only scans a flat `Lecture\*.pdf` directory, while Course 4186 is recursive and mixes `pdf`, `pptx`, `docx`, and one legacy `ppt`.
2. The old flow stores only chunk-level vectors. It does not maintain a dedicated knowledge-point layer, which is needed for structured exercise generation.
3. `D:\digital_human\retrival.py` expects `rag_answer()` to return a dict in one path, but `D:\digital_human\rag.py` currently returns a string. That mismatch would break reuse.
4. The current similarity filtering in `D:\digital_human\rag.py` uses `similarity_search_with_score()` and keeps results when `score >= 0.5`. For FAISS distance-based scores this is a risky assumption.
5. Prompting and quiz generation in the original project are tuned around a different course and are not specific enough for the 4186 lecture set.

Because of those issues, this folder uses a separate pipeline designed around the real structure of Course 4186.

## What the new pipeline does

The main script is `pipeline.py`. It performs the following steps:

1. Read the course materials from the configured course root.
2. Extract text from:
   - `pdf` via `pypdf`
   - `pptx` via Open XML parsing
   - `docx` via Open XML parsing
   - `ppt` via optional Windows COM fallback when available
3. Split extracted text into chunks for retrieval.
4. Build a chunk-level retriever.
5. Build a knowledge-point layer on top of the chunks.
6. Generate exercises from the retrieved supporting chunks.
7. Save all outputs under `artifacts\`.

## Retrieval design

This implementation uses a two-layer RAG layout:

1. Chunk RAG
   - Stores chunked lecture evidence.
   - Supports semantic retrieval with FAISS when embeddings are available.
   - Falls back to a lexical retriever if dense indexing is unavailable.
2. Knowledge-point RAG
   - Stores structured knowledge points as JSON plus an index.
   - Each knowledge point keeps source files, supporting chunk ids, keywords, and summaries.
   - Exercise generation retrieves support chunks from this layer before drafting questions.

## Outputs

The pipeline writes the following files under `artifacts\`:

- `inventory.json`
- `inventory.md`
- `raw_documents.jsonl`
- `chunks.jsonl`
- `knowledge_points.json`
- `questions.json`
- `chunk_index\...`
- `kp_index\...`

## Runtime environment

The verified environment for this folder is:

`C:\Users\langhuang6\AppData\Local\anaconda3\envs\env_meta\python.exe`

That environment already contains `pypdf`, `faiss`, `langchain`, `sentence_transformers`, and `openai`.

## Typical commands

Analyze the course materials:

```powershell
C:\Users\langhuang6\AppData\Local\anaconda3\envs\env_meta\python.exe D:\digital_human\course4186_rag\pipeline.py analyze
```

Build the artifacts without calling an LLM:

```powershell
C:\Users\langhuang6\AppData\Local\anaconda3\envs\env_meta\python.exe D:\digital_human\course4186_rag\pipeline.py build --no-llm
```

Ask a question against the built artifacts:

```powershell
C:\Users\langhuang6\AppData\Local\anaconda3\envs\env_meta\python.exe D:\digital_human\course4186_rag\pipeline.py ask --question "What is the role of epipolar geometry?"
```

## Optional LLM configuration

The script can optionally call an OpenAI-compatible endpoint when these environment variables are provided:

- `COURSE_LLM_API_KEY`
- `COURSE_LLM_BASE_URL`
- `COURSE_LLM_MODEL`

It also falls back to these names if they already exist in the current machine configuration:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_CHAT_MODEL`
- `KIMI_API_KEY`
- `KIMI_BASE_URL`
- `KIMI_MODEL`

If no valid LLM configuration is found, the script still works in deterministic fallback mode.

## Deployment note

For the current bundled web deployment, you do not need to rebuild the RAG data. The repository already includes the active `artifacts_full_course` bundle used by the latest portal.
