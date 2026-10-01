import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, class_names, detector, get_current_user
from app.services.users import DuplicateUserError


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_current_user] = lambda: {
            "id": 1,
            "username": "tester",
            "email": "tester@example.com",
            "full_name": None,
            "role": "user",
            "is_active": True,
        }
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_reports_api_and_missing_weights(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_configured": False})

    def test_lists_all_dataset_classes(self) -> None:
        response = self.client.get("/api/signs")
        self.assertEqual(response.status_code, 200)
        signs = response.json()
        self.assertEqual(len(signs), 56)
        self.assertEqual(signs[0], {"class_id": 0, "code": class_names[0]})
        self.assertEqual(signs[-1]["class_id"], 55)

    def test_detection_returns_service_unavailable_without_weights(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image_bytes, format="PNG")
        response = self.client.post(
            "/api/detect",
            files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Model weights were not found", response.json()["detail"])

    def test_detection_returns_class_code_and_pixel_box(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image_bytes, format="PNG")
        prediction = [
            {
                "class_id": 1,
                "confidence": 0.9,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 6.0,
                "y2": 7.0,
            }
        ]
        with (
            patch.object(detector, "predict", return_value=prediction),
            patch("app.main.save_image_detection", return_value=42),
        ):
            response = self.client.post(
                "/api/detect",
                files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["image_width"], 8)
        self.assertEqual(response.json()["history_id"], 42)
        self.assertEqual(response.json()["detections"][0]["class_id"], 1)
        self.assertEqual(response.json()["detections"][0]["code"], class_names[1])
        self.assertEqual(response.json()["detections"][0]["x2"], 6.0)

    def test_rejects_non_image_upload(self) -> None:
        response = self.client.post(
            "/api/detect",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 422)

    def test_detection_and_history_require_authentication(self) -> None:
        app.dependency_overrides.clear()
        response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 401)

    def test_register_returns_conflict_for_existing_account(self) -> None:
        with patch(
            "app.main.register_user",
            side_effect=DuplicateUserError("Username or email is already registered."),
        ):
            response = self.client.post(
                "/api/auth/register",
                json={
                    "username": "tester",
                    "email": "tester@example.com",
                    "password": "strong-password",
                },
            )
        self.assertEqual(response.status_code, 409)

    def test_login_returns_bearer_token(self) -> None:
        user = {
            "id": 1,
            "username": "tester",
            "email": "tester@example.com",
            "full_name": None,
            "role": "user",
            "is_active": True,
        }
        with (
            patch("app.main.authenticate_user", return_value=user),
            patch("app.main.create_access_token", return_value="signed-token"),
        ):
            response = self.client.post(
                "/api/auth/login",
                data={"username": "tester", "password": "strong-password"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"access_token": "signed-token", "token_type": "bearer"})

    def test_history_returns_only_authenticated_users_records(self) -> None:
        expected_history = [
            {
                "id": 5,
                "source_type": "image",
                "source_path": "test.png",
                "detected_at": "2026-09-30T00:00:00",
                "results": [
                    {
                        "class_id": 1,
                        "code": class_names[1],
                        "confidence": 0.9,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.25,
                        "height": 0.25,
                    }
                ],
            }
        ]
        with patch("app.main.list_user_detections", return_value=expected_history) as list_history:
            response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["id"], 5)
        list_history.assert_called_once_with(1, 20, 0)


if __name__ == "__main__":
    unittest.main()
