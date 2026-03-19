# Course 4186 Student Hub

This folder contains the standalone Flask portal for the Course 4186 student-facing system.

## Main files

- `app.py`: main Flask application
- `kb_service.py`: course-grounded retrieval and answer synthesis
- `progress_store.py`: local practice history store
- `run_4186_portal.py`: start the portal locally
- `run_public_4186_portal.py`: start the portal and open a Cloudflare tunnel

## Main routes

- `/chatapi_4186.html`
- `/quiz_4186`
- `/quiz_4186/practice/<kp_id>`
- `/quiz_4186/analysis`

## Local run

From the bundle root:

```powershell
python .\course4186_portal\run_4186_portal.py
```

Then open:

`http://127.0.0.1:50186/chatapi_4186.html`

## Public run

Make sure `cloudflared` is installed and available on `PATH`, or set `CLOUDFLARED_PATH`.

```powershell
python .\course4186_portal\run_public_4186_portal.py
```
