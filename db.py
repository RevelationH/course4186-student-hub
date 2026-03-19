from __future__ import annotations

import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


def _resolve_credential_path() -> Path:
    candidates = [
        os.getenv("FIREBASE_CREDENTIALS", "").strip(),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
        str(Path(__file__).resolve().parent / "firebase-service-account.json"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise RuntimeError(
        "Firebase credentials were not found. Set FIREBASE_CREDENTIALS or "
        "GOOGLE_APPLICATION_CREDENTIALS, or place firebase-service-account.json "
        "in the bundle root."
    )


if not firebase_admin._apps:
    credential_path = _resolve_credential_path()
    firebase_admin.initialize_app(credentials.Certificate(str(credential_path)))


class fire_db:
    def __init__(self):
        self.db = firestore.client()

    def collection(self, collection_name):
        return self.db.collection(collection_name)

    def collection_group(self, collection_name):
        return self.db.collection_group(collection_name)

    def document(self, collection_name, doc_name):
        return self.db.collection(collection_name).document(doc_name)

    def read_wq(self, collection_1, username, collection_2):
        return self.db.collection(collection_1).document(username).collection(collection_2)

    def read_doc(self, collection, username):
        return self.db.collection(collection).document(username).get()
