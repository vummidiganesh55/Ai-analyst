import json
from unittest.mock import MagicMock

import pytest

from app.storage.redis_manager import RedisManager


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def manager(mock_client):
    return RedisManager(
        client=mock_client,
        default_ttl=3600,
    )


def test_ping_success(manager, mock_client):
    mock_client.ping.return_value = True

    assert manager.ping() is True


def test_ping_failure(manager, mock_client):
    from redis.exceptions import ConnectionError

    mock_client.ping.side_effect = ConnectionError("Unavailable")

    assert manager.ping() is False


def test_set_json_value(manager, mock_client):
    mock_client.set.return_value = True

    result = manager.set(
        "test:key",
        {"revenue": 1000},
    )

    assert result is True

    mock_client.set.assert_called_once_with(
        "test:key",
        json.dumps({"revenue": 1000}),
        ex=3600,
    )


def test_get_json_value(manager, mock_client):
    mock_client.get.return_value = '{"revenue": 1000}'

    result = manager.get("test:key")

    assert result == {"revenue": 1000}


def test_get_missing_key(manager, mock_client):
    mock_client.get.return_value = None

    assert manager.get("missing:key") is None


def test_delete_existing_key(manager, mock_client):
    mock_client.delete.return_value = 1

    assert manager.delete("test:key") is True


def test_exists(manager, mock_client):
    mock_client.exists.return_value = 1

    assert manager.exists("test:key") is True


def test_set_custom_ttl(manager, mock_client):
    mock_client.set.return_value = True

    manager.set("test:key", {"value": 1}, ttl=120)

    mock_client.set.assert_called_once_with(
        "test:key",
        '{"value": 1}',
        ex=120,
    )


def test_invalid_ttl(manager):
    with pytest.raises(ValueError):
        manager.set("test:key", {"value": 1}, ttl=0)


def test_empty_key(manager):
    with pytest.raises(ValueError):
        manager.get("")


def test_non_string_key(manager):
    with pytest.raises(TypeError):
        manager.get(None)


def test_non_serializable_value(manager):
    with pytest.raises(ValueError):
        manager.set("test:key", {"value": object()})


def test_invalid_json(manager, mock_client):
    mock_client.get.return_value = "{invalid-json"

    with pytest.raises(ValueError):
        manager.get("test:key")


def test_set_session(manager, mock_client):
    mock_client.set.return_value = True

    context = {
        "session_id": "session-001",
        "messages": [],
    }

    assert manager.set_session("session-001", context)

    args, kwargs = mock_client.set.call_args

    assert args[0] == "bi:session:session-001"
    assert json.loads(args[1]) == context


def test_get_session(manager, mock_client):
    mock_client.get.return_value = json.dumps(
        {
            "session_id": "session-001",
            "messages": [],
        }
    )

    result = manager.get_session("session-001")

    assert result["session_id"] == "session-001"


def test_get_missing_session(manager, mock_client):
    mock_client.get.return_value = None

    assert manager.get_session("missing-session") is None


def test_delete_session(manager, mock_client):
    mock_client.delete.return_value = 1

    assert manager.delete_session("session-001")


def test_empty_session_id(manager):
    with pytest.raises(ValueError):
        manager.get_session("")


def test_cache_result(manager, mock_client):
    mock_client.set.return_value = True

    result = manager.cache_result(
        "sales-v1-revenue",
        {"revenue": 5000},
        ttl=300,
    )

    assert result is True

    args, kwargs = mock_client.set.call_args

    assert args[0] == "bi:cache:sales-v1-revenue"
    assert kwargs["ex"] == 300


def test_get_cached_result(manager, mock_client):
    mock_client.get.return_value = '{"revenue": 5000}'

    result = manager.get_cached_result(
        "sales-v1-revenue"
    )

    assert result == {"revenue": 5000}


def test_invalidate_cache(manager, mock_client):
    mock_client.delete.return_value = 1

    assert manager.invalidate_cache("sales-v1-revenue")


def test_redis_set_error_propagates(manager, mock_client):
    from redis.exceptions import ConnectionError

    mock_client.set.side_effect = ConnectionError(
        "Redis unavailable"
    )

    with pytest.raises(ConnectionError):
        manager.set("test:key", {"value": 1})


def test_redis_get_error_propagates(manager, mock_client):
    from redis.exceptions import ConnectionError

    mock_client.get.side_effect = ConnectionError(
        "Redis unavailable"
    )

    with pytest.raises(ConnectionError):
        manager.get("test:key")