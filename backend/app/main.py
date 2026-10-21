import base64
from io import BytesIO
import logging
import os
from pathlib import Path
import tempfile
import warnings

import cv2
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
import yaml

from app.config import DATA_CONFIG, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, MODEL_PATH, get_cors_origins
from app.schemas import (
    DetectionHistoryItem,
    HealthResponse,
    ImageDetectionResponse,
    RegisterRequest,
    SignResponse,
    TokenResponse,
    UserResponse,
)
from app.services.database import DatabaseUnavailableError
from app.services.detector import ModelUnavailableError, TrafficSignDetector
from app.services.history import list_user_detections, save_image_detection
from app.services.users import (
    AuthenticationUnavailableError,
    DuplicateUserError,
    InvalidCredentialsError,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_user,
    register_user,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="Traffic Sign Recognition API",
    description="YOLO-based traffic sign detection API.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

detector = TrafficSignDetector(MODEL_PATH)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def load_class_names() -> list[str]:
    if not DATA_CONFIG.is_file():
        raise RuntimeError(f"Dataset configuration file not found: {DATA_CONFIG}")
    with DATA_CONFIG.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise RuntimeError(f"Invalid class names in dataset configuration: {DATA_CONFIG}")
    return names


class_names = load_class_names()


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        user_id = decode_access_token(token)
        user = get_user(user_id)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (AuthenticationUnavailableError, DatabaseUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User account is unavailable.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_configured=detector.is_configured)


@app.get("/api/signs", response_model=list[SignResponse], tags=["traffic signs"])
def list_signs() -> list[SignResponse]:
    return [SignResponse(class_id=index, code=code) for index, code in enumerate(class_names)]


@app.post("/api/auth/register", response_model=UserResponse, status_code=201, tags=["authentication"])
def register(request: RegisterRequest) -> UserResponse:
    try:
        user = register_user(
            username=request.username,
            email=str(request.email).lower(),
            password=request.password,
            full_name=request.full_name,
        )
    except DuplicateUserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return UserResponse(**user)


@app.post("/api/auth/login", response_model=TokenResponse, tags=["authentication"])
def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    try:
        user = authenticate_user(form.username, form.password)
        token = create_access_token(user["id"])
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (AuthenticationUnavailableError, DatabaseUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserResponse, tags=["authentication"])
def current_user(user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse(**user)


@app.get("/api/history", response_model=list[DetectionHistoryItem], tags=["history"])
def detection_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> list[DetectionHistoryItem]:
    try:
        history = list_user_detections(user["id"], limit, offset)
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [DetectionHistoryItem(**item) for item in history]


def is_video_filename_or_mime(filename: str, content_type: str | None) -> bool:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    ext = Path(filename).suffix.lower() if filename else ""
    return ext in video_exts or (content_type is not None and content_type.startswith("video/"))


@app.post("/api/detect", response_model=ImageDetectionResponse, tags=["recognition"])
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0, le=1),
    user: dict = Depends(get_current_user),
) -> ImageDetectionResponse:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    filename = file.filename or "upload"
    is_video = is_video_filename_or_mime(filename, file.content_type)

    if is_video:
        # Video processing with OpenCV
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=422, detail="Could not read the uploaded video file.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0 or video_w <= 0 or video_h <= 0:
                raise HTTPException(status_code=422, detail="Invalid video frame dimensions.")

            num_samples = min(max(total_frames, 1), 30)
            step = max(1, total_frames // num_samples)

            best_frame = None
            best_detections: list[dict] = []
            best_max_conf = -1.0

            for frame_idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                if best_frame is None:
                    best_frame = frame

                try:
                    frame_dets = detector.predict(frame, confidence)
                except ModelUnavailableError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc

                max_conf = max([d["confidence"] for d in frame_dets], default=0.0)
                if max_conf > best_max_conf or not best_detections:
                    best_max_conf = max_conf
                    best_detections = frame_dets
                    best_frame = frame
                    if max_conf >= 0.85:
                        break

            cap.release()
            if best_frame is None:
                raise HTTPException(status_code=422, detail="Could not extract any valid frame from the video.")

            image_height, image_width = best_frame.shape[:2]
            detections = best_detections

            _, buf = cv2.imencode(".jpg", best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            image_data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
            source_type = "video"
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    else:
        # Image processing
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(contents)) as uploaded_image:
                    uploaded_image.verify()
                with Image.open(BytesIO(contents)) as uploaded_image:
                    image_width, image_height = uploaded_image.size
                    image = uploaded_image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise HTTPException(status_code=422, detail="The uploaded file is not a valid, safe image.") from exc
        finally:
            await file.close()

        image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        try:
            detections = detector.predict(image_bgr, confidence)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        _, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        source_type = "image"

    for detection in detections:
        class_id = int(detection["class_id"])
        if class_id < 0 or class_id >= len(class_names):
            logger.error("Model returned class id %d outside the configured class range", class_id)
            raise HTTPException(status_code=500, detail="Model class IDs do not match the dataset configuration.")
        detection["code"] = class_names[class_id]

    try:
        history_id = save_image_detection(
            user_id=user["id"],
            filename=filename,
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            source_type=source_type,
        )
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ImageDetectionResponse(
        image_width=image_width,
        image_height=image_height,
        history_id=history_id,
        detections=detections,
        image_data=image_data_url,
    )
