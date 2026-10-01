import logging
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.services.database import DatabaseUnavailableError, get_engine

logger = logging.getLogger(__name__)


def save_image_detection(
    user_id: int,
    filename: str,
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> int:
    safe_filename = Path(filename).name[:255] if filename else "upload"
    try:
        with get_engine().begin() as connection:
            model_id = connection.execute(
                text("SELECT id FROM models WHERE is_active = TRUE ORDER BY id DESC LIMIT 1")
            ).scalar_one_or_none()
            result = connection.execute(
                text(
                    "INSERT INTO detections (user_id, model_id, source_type, source_path) "
                    "VALUES (:user_id, :model_id, 'image', :source_path)"
                ),
                {"user_id": user_id, "model_id": model_id, "source_path": safe_filename},
            )
            detection_id = result.lastrowid
            for detection in detections:
                sign_id = connection.execute(
                    text("SELECT id FROM traffic_signs WHERE class_id = :class_id"),
                    {"class_id": detection["class_id"]},
                ).scalar_one_or_none()
                if sign_id is None:
                    raise DatabaseUnavailableError(
                        f"Traffic sign class {detection['class_id']} is not seeded in the database."
                    )
                x1, y1, x2, y2 = (
                    detection["x1"],
                    detection["y1"],
                    detection["x2"],
                    detection["y2"],
                )
                connection.execute(
                    text(
                        "INSERT INTO detection_details "
                        "(detection_id, traffic_sign_id, confidence, x_center, y_center, width, height) "
                        "VALUES (:detection_id, :traffic_sign_id, :confidence, :x_center, :y_center, :width, :height)"
                    ),
                    {
                        "detection_id": detection_id,
                        "traffic_sign_id": sign_id,
                        "confidence": detection["confidence"],
                        "x_center": ((x1 + x2) / 2) / image_width,
                        "y_center": ((y1 + y2) / 2) / image_height,
                        "width": (x2 - x1) / image_width,
                        "height": (y2 - y1) / image_height,
                    },
                )
        return int(detection_id)
    except DatabaseUnavailableError:
        raise
    except SQLAlchemyError as exc:
        logger.exception("Could not save detection history")
        raise DatabaseUnavailableError("Could not save detection history because the database is unavailable.") from exc


def list_user_detections(user_id: int, limit: int, offset: int) -> list[dict[str, Any]]:
    try:
        with get_engine().connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT d.id, d.source_type, d.source_path, d.detected_at, "
                    "dd.confidence, ts.class_id, ts.code, dd.x_center, dd.y_center, dd.width, dd.height "
                    "FROM detections d "
                    "LEFT JOIN detection_details dd ON dd.detection_id = d.id "
                    "LEFT JOIN traffic_signs ts ON ts.id = dd.traffic_sign_id "
                    "WHERE d.user_id = :user_id "
                    "ORDER BY d.detected_at DESC, d.id DESC "
                    "LIMIT :limit OFFSET :offset"
                ),
                {"user_id": user_id, "limit": limit, "offset": offset},
            ).mappings().all()
    except SQLAlchemyError as exc:
        logger.exception("Could not read detection history")
        raise DatabaseUnavailableError("Could not read detection history because the database is unavailable.") from exc

    history: dict[int, dict[str, Any]] = {}
    for row in rows:
        detection = history.setdefault(
            row["id"],
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "source_path": row["source_path"],
                "detected_at": row["detected_at"],
                "results": [],
            },
        )
        if row["class_id"] is not None:
            detection["results"].append(
                {
                    "class_id": row["class_id"],
                    "code": row["code"],
                    "confidence": row["confidence"],
                    "x_center": row["x_center"],
                    "y_center": row["y_center"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
    return list(history.values())
