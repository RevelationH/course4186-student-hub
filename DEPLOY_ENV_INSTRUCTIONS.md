# Course 4186 deployment env

Use `deploy.env` at the repository root when deploying the latest Course 4186 student hub.

## What this file does

- Enables the LLM-backed Course 4186 features with the current Kimi/Moonshot endpoint.
- Does not override existing server environment variables because the app loads env files with `override=False`.
- Keeps Firebase / Firestore configuration optional in the env file, so an existing server-side Firebase setup remains unchanged.

## Recommended deployment steps

1. Put `deploy.env` in the repository root, or copy `.env.example` to `.env` and fill in the values.
2. If the server already has `FIREBASE_CREDENTIALS` or `GOOGLE_APPLICATION_CREDENTIALS` configured, leave the Firebase lines in `deploy.env` empty.
3. If the server does not already have Firebase configured, set one of these in `deploy.env`:
   - `FIREBASE_CREDENTIALS=/absolute/path/to/firebase-service-account.json`
   - or `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/firebase-service-account.json`
4. For a bundle-clone deployment, point the portal to the bundled materials and bundled full-course artifacts:
   - `COURSE4186_MATERIAL_ROOT=/absolute/path/to/course4186_materials`
   - `COURSE4186_ARTIFACT_DIR=/absolute/path/to/course4186_rag/artifacts_full_course`
5. Start the app normally:

```bash
python ./course4186_portal/run_4186_portal.py
```

## Notes

- The latest project defaults to Kimi / Moonshot through `COURSE_LLM_*`.
- `deploy.env` is loaded automatically by the project.
- Existing system environment variables still have higher priority than values inside `deploy.env`.
- The current public-facing bundle is the full-course `Week 1` to `Week 12` version.
