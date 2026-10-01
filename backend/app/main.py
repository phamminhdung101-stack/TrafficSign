from io import BytesIO
import logging
import warnings

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


@app.post("/api/detect", response_model=ImageDetectionResponse, tags=["recognition"])
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0, le=1),
    user: dict = Depends(get_current_user),
) -> ImageDetectionResponse:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

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

    try:
        detections = detector.predict(np.asarray(image), confidence)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for detection in detections:
        class_id = int(detection["class_id"])
        if class_id < 0 or class_id >= len(class_names):
            logger.error("Model returned class id %d outside the configured class range", class_id)
            raise HTTPException(status_code=500, detail="Model class IDs do not match the dataset configuration.")
        detection["code"] = class_names[class_id]

    try:
        history_id = save_image_detection(
            user_id=user["id"],
            filename=file.filename or "upload",
            detections=detections,
            image_width=image_width,
            image_height=image_height,
        )
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ImageDetectionResponse(
        image_width=image_width,
        image_height=image_height,
        history_id=history_id,
        detections=detections,
    )
