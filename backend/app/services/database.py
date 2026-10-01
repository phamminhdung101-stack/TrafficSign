import logging
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _create_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_engine() -> Engine:
    if not DATABASE_URL:
        raise DatabaseUnavailableError("DATABASE_URL is not configured.")
    try:
        return _create_engine(DATABASE_URL)
    except SQLAlchemyError as exc:
        logger.exception("Could not initialize the database engine")
        raise DatabaseUnavailableError("The database is unavailable.") from exc
