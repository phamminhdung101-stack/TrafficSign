import os
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text

from app.services import database
from app.services.history import list_user_detections, save_image_detection
from app.services.users import (
    DuplicateUserError,
    InvalidCredentialsError,
    authenticate_user,
    create_access_token,
    decode_access_token,
    register_user,
)
from app.main import class_names


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        handle, self.database_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(handle)
        self.database_url = f"sqlite:///{self.database_path.replace(os.sep, '/')}"
        self.engine = create_engine(self.database_url)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, "
                    "email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, full_name TEXT, "
                    "role TEXT NOT NULL, is_active BOOLEAN NOT NULL)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE traffic_signs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "class_id INTEGER UNIQUE NOT NULL, code TEXT UNIQUE NOT NULL)"
                )
            )
            connection.execute(
                text("CREATE TABLE models (id INTEGER PRIMARY KEY AUTOINCREMENT, is_active BOOLEAN NOT NULL)")
            )
            connection.execute(
                text(
                    "CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, "
                    "model_id INTEGER, source_type TEXT NOT NULL, source_path TEXT, "
                    "detected_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE detection_details (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "detection_id INTEGER NOT NULL, traffic_sign_id INTEGER NOT NULL, confidence FLOAT NOT NULL, "
                    "x_center FLOAT, y_center FLOAT, width FLOAT, height FLOAT)"
                )
            )
            connection.execute(
                text("INSERT INTO traffic_signs (class_id, code) VALUES (:class_id, :code)"),
                [{"class_id": index, "code": code} for index, code in enumerate(class_names)],
            )
        self.database_patch = patch.object(database, "DATABASE_URL", self.database_url)
        self.database_patch.start()

    def tearDown(self) -> None:
        self.database_patch.stop()
        database._create_engine(self.database_url).dispose()
        database._create_engine.cache_clear()
        self.engine.dispose()
        os.unlink(self.database_path)

    def test_register_login_and_jwt(self) -> None:
        user = register_user("tester", "tester@example.com", "strong-password", "Test User")
        self.assertEqual(user["id"], 1)
        self.assertEqual(authenticate_user("tester", "strong-password")["email"], "tester@example.com")
        with self.assertRaises(InvalidCredentialsError):
            authenticate_user("tester", "wrong-password")
        with self.assertRaises(DuplicateUserError):
            register_user("tester", "another@example.com", "strong-password", None)
        with patch("app.services.users.JWT_SECRET_KEY", "a" * 32):
            token = create_access_token(user["id"])
            self.assertEqual(decode_access_token(token), user["id"])

    def test_detection_history_is_saved_with_normalized_boxes_and_user_scope(self) -> None:
        user = register_user("tester", "tester@example.com", "strong-password", None)
        detections = [
            {
                "class_id": 1,
                "confidence": 0.9,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 60.0,
            }
        ]
        history_id = save_image_detection(user["id"], "..\\private\\sample.png", detections, 100, 200)
        history = list_user_detections(user["id"], 20, 0)
        self.assertEqual(history[0]["id"], history_id)
        self.assertEqual(history[0]["source_path"], "sample.png")
        self.assertEqual(history[0]["results"][0]["code"], class_names[1])
        self.assertAlmostEqual(history[0]["results"][0]["x_center"], 0.2)
        self.assertAlmostEqual(history[0]["results"][0]["y_center"], 0.2)
        self.assertAlmostEqual(history[0]["results"][0]["width"], 0.2)
        self.assertAlmostEqual(history[0]["results"][0]["height"], 0.2)
        self.assertEqual(list_user_detections(user["id"] + 1, 20, 0), [])


if __name__ == "__main__":
    unittest.main()
