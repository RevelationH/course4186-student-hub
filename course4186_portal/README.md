# Course 4186 Student Hub

This folder contains a standalone student-facing chat, quiz, and learning-report hub for Course 4186.

It does not modify or replace the original files under:

- `D:\digital_human\web`
- `D:\digital_human\templates`
- `D:\digital_human\quiz_app.py`
- `D:\digital_human\app.py`

Instead, it reads the new knowledge-base artifacts built in:

- `D:\digital_human\course4186_rag\artifacts`
- `D:\digital_human\course4186_rag\artifacts_dense`

## Main files

- `app.py`: standalone Flask app for the student hub
- `kb_service.py`: retrieval, course-grounded answer assembly, question loading
- `progress_store.py`: local practice-progress storage

## Main routes

- `/chatapi_4186.html`
- `/quiz_4186`
- `/quiz_4186/practice/<kp_id>`
- `/quiz_4186/analysis`

## Run

```powershell
C:\Users\langhuang6\AppData\Local\anaconda3\envs\env_meta\python.exe D:\digital_human\course4186_portal\app.py --host 0.0.0.0 --port 50186
```

Then open:

`http://127.0.0.1:50186/chatapi_4186.html`
