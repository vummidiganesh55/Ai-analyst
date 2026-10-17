from datetime import datetime

import pytest

from app.integrations.mcp_server import (
    SERVICE_NAME,
    create_mcp_server,
    health_check,
)


def test_health_check_returns_healthy_status():
    result = health_check()

    assert result["status"] == "healthy"


def test_health_check_returns_service_name():
    result = health_check()

    assert result["service"] == SERVICE_NAME


def test_health_check_returns_version():
    result = health_check()

    assert result["version"] == "1.0.0"


def test_health_check_returns_timestamp():
    result = health_check()

    timestamp = datetime.fromisoformat(result["timestamp"])

    assert timestamp.tzinfo is not None


def test_health_check_returns_dictionary():
    result = health_check()

    assert isinstance(result, dict)


def test_health_check_contains_required_fields():
    result = health_check()

    required_fields = {
        "status",
        "service",
        "timestamp",
        "version",
    }

    assert required_fields.issubset(result.keys())


def test_service_name_is_not_empty():
    assert SERVICE_NAME
    assert isinstance(SERVICE_NAME, str)


def test_create_mcp_server_returns_server():
    server = create_mcp_server()

    assert isinstance(server, type(create_mcp_server()))


def test_repeated_server_creation():
    first_server = create_mcp_server()
    second_server = create_mcp_server()

    assert first_server is not second_server