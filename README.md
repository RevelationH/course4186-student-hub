# Course 4186 GitHub Bundle

This folder is the latest self-contained bundle of the Course 4186 student system.

## Included

- `course4186_portal/`: Flask portal for chat, quiz, and learning report
- `course4186_rag/`: RAG pipeline plus the bundled full-course artifacts
- `course4186_materials/`: canonical lecture PDFs copied for server deployment
- `web/c++.jpg`: image asset used by the classic view
- `db.py` and `user.py`: Firebase-backed login integration
- `kimi_utils.py`: environment-based Kimi configuration only

## Secrets

No Firebase service-account JSON or hardcoded API key is included in this bundle.

Before running, configure:

- `FIREBASE_CREDENTIALS` or `GOOGLE_APPLICATION_CREDENTIALS`
- `COURSE_LLM_API_KEY`
- `COURSE_LLM_BASE_URL`
- `COURSE_LLM_MODEL`

Optional compatibility variables:

- `KIMI_API_KEY`
- `KIMI_API_BASE`
- `KIMI_MODEL`
- `COURSE4186_ARTIFACT_DIR`
- `COURSE4186_COURSE_ROOT`
- `CLOUDFLARED_PATH`

## Firebase Setup

The portal uses Firebase for student login and chat-session storage.

On Linux, export one of the following before starting the portal:

```bash
export FIREBASE_CREDENTIALS=/home/ubuntu/firebase-service-account.json
```

or

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/home/ubuntu/firebase-service-account.json
```

The code will look for credentials in this order:

1. `FIREBASE_CREDENTIALS`
2. `GOOGLE_APPLICATION_CREDENTIALS`
3. `firebase-service-account.json` in the repository root

If none of these exist, startup will stop with a clear Firebase configuration error.

## Install

```powershell
pip install -r .\requirements.txt
```

## Run locally

```powershell
python .\course4186_portal\run_4186_portal.py
```

Open:

`http://127.0.0.1:50186/chatapi_4186.html`

## Run with public access

Install Cloudflare Tunnel first, then run:

```powershell
python .\course4186_portal\run_public_4186_portal.py
```

## Notes

- The active bundled RAG data is the `Week 1` to `Week 12` full-course version under `course4186_rag/artifacts_full_course`.
- Lecture source buttons are designed to open browser-friendly files. The bundle therefore includes canonical lecture PDFs in `course4186_materials/`.
- If a citation source in the artifacts points to a `pptx`, the portal will still resolve it to the matching PDF when possible.
- The portal stores local study progress in `course4186_portal/data/progress.json`.
- Runtime logs are written to `course4186_portal/logs/`.
