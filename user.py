from __future__ import annotations

from db import fire_db


class User:
    def __init__(self, username, password, is_admin=False):
        self.username = username
        self.password = password
        self.is_admin = is_admin
        self.fdb = fire_db()

    def save(self):
        self.fdb.collection("users").document(self.username).set(
            {
                "username": self.username,
                "password": self.password,
                "is_admin": self.is_admin,
            }
        )

    @classmethod
    def get_by_username(cls, username):
        temp_user = cls(username, "")
        doc = temp_user.fdb.read_doc("users", username)
        if not doc.exists:
            return None
        data = doc.to_dict()
        return cls(data["username"], data["password"], data.get("is_admin", False))

    def add_wrong_answer(self, question, std_answer, user_answer, timestamp, keypoint):
        wrong_answer_data = {
            "question": question,
            "std_answer": std_answer,
            "user_answer": user_answer,
            "timestamp": timestamp,
        }

        self.fdb.collection("users").document(self.username).collection("wrong_questions").document(keypoint).collection(
            "questions"
        ).add(wrong_answer_data)
