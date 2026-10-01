from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class HealthResponse(BaseModel):
    status: str
    model_configured: bool


class SignResponse(BaseModel):
    class_id: int
    code: str


class DetectionResponse(BaseModel):
    class_id: int
    code: str
    confidence: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float


class ImageDetectionResponse(BaseModel):
    image_width: int
    image_height: int
    history_id: int
    detections: list[DetectionResponse]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=100)


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    full_name: str | None
    role: Literal["user", "admin"]
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class DetectionHistoryResult(BaseModel):
    class_id: int
    code: str
    confidence: float = Field(ge=0, le=1)
    x_center: float
    y_center: float
    width: float
    height: float


class DetectionHistoryItem(BaseModel):
    id: int
    source_type: Literal["image", "video", "camera"]
    source_path: str | None
    detected_at: datetime
    results: list[DetectionHistoryResult]
