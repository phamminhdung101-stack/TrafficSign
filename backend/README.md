# Traffic Sign Recognition API

FastAPI backend for the 56-class YOLO traffic-sign dataset in `dataset/data.yaml`.

## Setup

From the repository root, create and activate a virtual environment, then install the backend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Place trained Ultralytics YOLO weights at `backend/models/best.pt`, or set `TRAFFIC_SIGN_MODEL` to the weight file. The repository does not include trained weights. `TRAFFIC_SIGN_DATA` can override the dataset YAML path.

## Database and authentication

Create the MySQL database and tables by running `database/db_traffic.sql`, then configure `DATABASE_URL` and `JWT_SECRET_KEY` in the environment. Use a random JWT secret of at least 32 bytes; for example, generate one with `py -c "import secrets; print(secrets.token_urlsafe(48))"`. Do not use the example secret in a deployed environment.

The API does not create database tables automatically. Register a user through `POST /api/auth/register`, then log in through `POST /api/auth/login` using form fields `username` and `password`. Registration creates ordinary user accounts; it never grants administrator privileges.

To promote a trusted registered account, run `UPDATE users SET role = 'admin' WHERE username = 'your-username';` in MySQL. The SQL setup script intentionally does not insert a default account with a known plaintext password.

## Run

From the repository root:

```powershell
uvicorn app.main:app --app-dir backend --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /api/health` — API status and whether model weights are present.
- `GET /api/signs` — class IDs and sign codes loaded from `dataset/data.yaml`.
- `POST /api/auth/register` — create an account with username, email, password, and optional full name.
- `POST /api/auth/login` — exchange OAuth2 form credentials for a bearer token.
- `GET /api/auth/me` — return the authenticated account profile.
- `POST /api/detect?confidence=0.25` — authenticated multipart image upload with a `file` field. Returns image dimensions, history ID, class ID, sign code, confidence, and pixel bounding boxes. Saves the detection and normalized bounding boxes to MySQL.
- `GET /api/history?limit=20&offset=0` — return only the authenticated user's detection history, with normalized bounding boxes.

Image uploads are limited to 10 MiB by default. Configure `MAX_UPLOAD_BYTES`, `MAX_IMAGE_PIXELS`, and `CORS_ORIGINS` through environment variables as needed. Detection returns HTTP 503 until valid trained weights and a configured database are available. Seed `traffic_signs` using the SQL script so predicted classes can be saved to detection history.
