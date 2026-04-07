import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

from env_loader import load_project_env

load_project_env()


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CERT_NAMES = (
    "firebase-service-account.json",
    "quizsite-fb97c-firebase-adminsdk-fbsvc-76a794e54f.json",
)


def _load_certificate_from_value(raw_value: str):
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("Empty Firebase credential value.")

    if value.startswith("{"):
        try:
            return credentials.Certificate(json.loads(value))
        except json.JSONDecodeError as exc:
            raise RuntimeError("FIREBASE_CREDENTIALS looks like JSON, but it is not valid JSON.") from exc

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Firebase credential file was not found: {path}")

    return credentials.Certificate(str(path))


def _resolve_default_certificate():
    for filename in DEFAULT_CERT_NAMES:
        candidate = ROOT_DIR / filename
        if candidate.exists():
            return credentials.Certificate(str(candidate))
    return None


def _initialize_firebase():
    if firebase_admin._apps:
        return

    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS", "").strip()
    google_app_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if firebase_credentials:
        try:
            firebase_admin.initialize_app(_load_certificate_from_value(firebase_credentials))
            return
        except Exception as exc:
            raise RuntimeError(
                "Firebase initialization failed. "
                "Check FIREBASE_CREDENTIALS and make sure it points to a valid service-account JSON file "
                "or contains valid JSON credentials."
            ) from exc

    if google_app_credentials:
        try:
            firebase_admin.initialize_app(_load_certificate_from_value(google_app_credentials))
            return
        except Exception as exc:
            raise RuntimeError(
                "Firebase initialization failed. "
                "Check GOOGLE_APPLICATION_CREDENTIALS and make sure it points to a valid service-account JSON file."
            ) from exc

    fallback_certificate = _resolve_default_certificate()
    if fallback_certificate is not None:
        firebase_admin.initialize_app(fallback_certificate)
        return

    raise RuntimeError(
        "Firebase credentials are not configured. "
        "Set FIREBASE_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS to your Firebase service-account JSON file path, "
        "or place firebase-service-account.json in the repository root."
    )


_initialize_firebase()


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
