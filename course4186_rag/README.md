# Course 4186 RAG Pipeline

This folder contains the standalone RAG build pipeline used by the Course 4186 student hub.

## Scope

- The bundled artifacts already cover `Week 1` to `Week 6`.
- `pipeline.py` now defaults to:
  - `artifacts_week1_week6` as the output folder
  - `6` as the default `--max-week`

## What it does

`pipeline.py` can:

1. Read lecture and tutorial files recursively.
2. Extract text from `pdf`, `pptx`, `docx`, and legacy `ppt` files.
3. Build chunk-level retrieval data.
4. Build a knowledge-point layer on top of the chunks.
5. Generate multiple-choice practice questions grounded in the course materials.

## Main outputs

- `artifacts_week1_week6/knowledge_points.json`
- `artifacts_week1_week6/questions.json`
- `artifacts_week1_week6/chunks.jsonl`
- `artifacts_week1_week6/slide_images.json`
- `artifacts_week1_week6/chunk_index/...`
- `artifacts_week1_week6/kp_index/...`

## Typical commands

Analyze the source materials:

```powershell
python .\course4186_rag\pipeline.py analyze --course-root "D:\digital_human\4186\4186"
```

Rebuild the bundled Week 1 to Week 6 artifacts without calling an LLM:

```powershell
python .\course4186_rag\pipeline.py build --course-root "D:\digital_human\4186\4186" --no-llm
```

Ask a question against the bundled artifacts:

```powershell
python .\course4186_rag\pipeline.py ask --question "What is epipolar geometry?"
```

## Optional environment variables

- `COURSE4186_COURSE_ROOT`
- `COURSE4186_MAX_WEEK`
- `COURSE_LLM_API_KEY`
- `COURSE_LLM_BASE_URL`
- `COURSE_LLM_MODEL`
- `KIMI_API_KEY`
- `KIMI_API_BASE`
- `KIMI_MODEL`
