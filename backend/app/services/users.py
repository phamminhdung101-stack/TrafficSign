import logging

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET_KEY
from app.services.database import DatabaseUnavailableError, get_engine

logger = logging.getLogger(__name__)
password_hash = PasswordHash.recommended()


class DuplicateUserError(RuntimeError):
    pass


class InvalidCredentialsError(RuntimeError):
    pass


class AuthenticationUnavailableError(RuntimeError):
    pass


def require_authentication_config() -> None:
    if len(JWT_SECRET_KEY.encode("utf-8")) < 32:
        raise AuthenticationUnavailableError("JWT_SECRET_KEY must contain at least 32 bytes.")


def create_access_token(user_id: int) -> str:
    require_authentication_config()
    from datetime import datetime, timedelta, timezone

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": str(user_id), "exp": expires_at}, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    require_authentication_config()
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        subject = payload.get("sub")
        if not isinstance(subject, str):
            raise InvalidCredentialsError("Invalid access token.")
        return int(subject)
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise InvalidCredentialsError("Invalid or expired access token.") from exc


def register_user(username: str, email: str, password: str, full_name: str | None) -> dict:
    hashed_password = password_hash.hash(password)
    try:
        with get_engine().begin() as connection:
            result = connection.execute(
                text(
                    "INSERT INTO users (username, email, password_hash, full_name, role, is_active) "
                    "VALUES (:username, :email, :password_hash, :full_name, 'user', TRUE)"
                ),
                {
                    "username": username,
                    "email": email,
                    "password_hash": hashed_password,
                    "full_name": full_name,
                },
            )
            user_id = result.lastrowid
            user = connection.execute(
                text(
                    "SELECT id, username, email, full_name, role, is_active "
                    "FROM users WHERE id = :user_id"
                ),
                {"user_id": user_id},
            ).mappings().one()
            return dict(user)
    except IntegrityError as exc:
        raise DuplicateUserError("Username or email is already registered.") from exc
    except SQLAlchemyError as exc:
        logger.exception("Could not register user")
        raise DatabaseUnavailableError("Could not register the user because the database is unavailable.") from exc


def authenticate_user(username: str, password: str) -> dict:
    try:
        with get_engine().connect() as connection:
            user = connection.execute(
                text(
                    "SELECT id, username, email, full_name, role, is_active, password_hash "
                    "FROM users WHERE username = :username"
                ),
                {"username": username},
            ).mappings().first()
    except SQLAlchemyError as exc:
        logger.exception("Could not authenticate user")
        raise DatabaseUnavailableError("Could not authenticate because the database is unavailable.") from exc

    try:
        valid_password = user is not None and password_hash.verify(password, user["password_hash"])
    except UnknownHashError:
        valid_password = False
    if not valid_password or not user["is_active"]:
        raise InvalidCredentialsError("Incorrect username or password.")
    return dict(user)


def get_user(user_id: int) -> dict | None:
    try:
        with get_engine().connect() as connection:
            user = connection.execute(
                text(
                    "SELECT id, username, email, full_name, role, is_active "
                    "FROM users WHERE id = :user_id AND is_active = TRUE"
                ),
                {"user_id": user_id},
            ).mappings().first()
            return dict(user) if user else None
    except SQLAlchemyError as exc:
        logger.exception("Could not read user profile")
        raise DatabaseUnavailableError("Could not read the user profile because the database is unavailable.") from exc
