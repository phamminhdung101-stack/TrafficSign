import logging
from pathlib import Path
from threading import Lock
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class TrafficSignDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Any = None
        self._lock = Lock()

    @property
    def is_configured(self) -> bool:
        return self.model_path.is_file()

    def reload(self) -> None:
        """Clear cached model to force reloading on the next predict call."""
        with self._lock:
            self._model = None

    def predict(self, image: Union[np.ndarray, Image.Image], confidence: float) -> list[dict[str, Any]]:
        # Ensure BGR numpy array: Ultralytics expects BGR uint8 arrays and flips to RGB internally
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif not isinstance(image, np.ndarray):
            image = np.asarray(image)

        with self._lock:
            model = self._load_model()
            try:
                results = model.predict(source=image, conf=confidence, verbose=False)
            except Exception as exc:
                logger.exception("YOLO inference failed")
                raise ModelUnavailableError("The configured model failed during inference.") from exc

        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": float(box.conf[0].item()),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        return detections

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.is_configured:
            raise ModelUnavailableError(
                f"Model weights were not found at {self.model_path}. "
                "Set TRAFFIC_SIGN_MODEL to a trained .pt file."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            logger.exception("Could not load YOLO model from %s", self.model_path)
            raise ModelUnavailableError("The configured YOLO model could not be loaded.") from exc
        return self._model
