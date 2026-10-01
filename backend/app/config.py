import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = Path(os.getenv("TRAFFIC_SIGN_DATA", PROJECT_ROOT / "dataset" / "data.yaml"))
MODEL_PATH = Path(os.getenv("TRAFFIC_SIGN_MODEL", PROJECT_ROOT / "backend" / "models" / "best.pt"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))


def get_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5500")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]
