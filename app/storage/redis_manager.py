from __future__ import annotations

import json
import logging
from typing import Any

import redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


class RedisManager:
    """
    Redis storage manager for conversation sessions and analytics caching.

    Features:
    - JSON serialization
    - Key expiration
    - Session persistence
    - Analytics result caching
    - Delete operations
    - Controlled error handling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        default_ttl: int = 3600,
        client: Any | None = None,
        socket_timeout: float = 3.0,
    ):
        if default_ttl <= 0:
            raise ValueError("default_ttl must be greater than zero")

        if socket_timeout <= 0:
            raise ValueError("socket_timeout must be greater than zero")

        self.default_ttl = default_ttl

        self.client = client or redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=socket_timeout,
            socket_timeout=socket_timeout,
        )

    def ping(self) -> bool:
        """Check whether Redis is reachable."""
        try:
            return bool(self.client.ping())
        except RedisError:
            logger.exception("Redis ping failed")
            return False

    @staticmethod
    def _serialize(value: Any) -> str:
        """Convert a Python value to JSON."""
        try:
            return json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Value is not JSON serializable: {exc}"
            ) from exc

    @staticmethod
    def _deserialize(value: str | None) -> Any:
        """Convert JSON into a Python value."""
        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Stored value contains invalid JSON: {exc}"
            ) from exc

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store a JSON value with an expiration time."""
        self._validate_key(key)

        effective_ttl = (
            self.default_ttl if ttl is None else ttl
        )

        if effective_ttl <= 0:
            raise ValueError("ttl must be greater than zero")

        serialized = self._serialize(value)

        try:
            result = self.client.set(
                key,
                serialized,
                ex=effective_ttl,
            )
            return bool(result)
        except RedisError:
            logger.exception("Redis set failed for key %s", key)
            raise

    def get(self, key: str) -> Any:
        """Retrieve and deserialize a JSON value."""
        self._validate_key(key)

        try:
            value = self.client.get(key)
            return self._deserialize(value)
        except RedisError:
            logger.exception("Redis get failed for key %s", key)
            raise

    def delete(self, key: str) -> bool:
        """Delete a key. Return True if it existed."""
        self._validate_key(key)

        try:
            return bool(self.client.delete(key))
        except RedisError:
            logger.exception("Redis delete failed for key %s", key)
            raise

    def exists(self, key: str) -> bool:
        """Check whether a key exists."""
        self._validate_key(key)

        try:
            return bool(self.client.exists(key))
        except RedisError:
            logger.exception("Redis exists failed for key %s", key)
            raise

    def set_session(
        self,
        session_id: str,
        context: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Persist a conversation context under its session ID."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        key = f"bi:session:{session_id.strip()}"

        return self.set(key, context, ttl=ttl)

    def get_session(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a persisted conversation context."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        result = self.get(
            f"bi:session:{session_id.strip()}"
        )

        if result is None:
            return None

        if not isinstance(result, dict):
            raise ValueError("Stored session context must be a JSON object")

        return result

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        return self.delete(
            f"bi:session:{session_id.strip()}"
        )

    def cache_result(
        self,
        cache_key: str,
        result: Any,
        ttl: int = 300,
    ) -> bool:
        """Cache an analytics result."""
        if not isinstance(cache_key, str) or not cache_key.strip():
            raise ValueError("cache_key cannot be empty")

        if ttl <= 0:
            raise ValueError("ttl must be greater than zero")

        return self.set(
            f"bi:cache:{cache_key.strip()}",
            result,
            ttl=ttl,
        )

    def get_cached_result(
        self,
        cache_key: str,
    ) -> Any:
        """Retrieve a cached analytics result."""
        if not isinstance(cache_key, str) or not cache_key.strip():
            raise ValueError("cache_key cannot be empty")

        return self.get(
            f"bi:cache:{cache_key.strip()}"
        )

    def invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate one cached result."""
        if not isinstance(cache_key, str) or not cache_key.strip():
            raise ValueError("cache_key cannot be empty")

        return self.delete(
            f"bi:cache:{cache_key.strip()}"
        )

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError("key must be a string")

        if not key.strip():
            raise ValueError("key cannot be empty")