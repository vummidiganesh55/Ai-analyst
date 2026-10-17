import pytest

from app.agents.self_correction import (
    CorrectionResult,
    SelfCorrectionEngine,
)


def test_successful_execution():
    engine = SelfCorrectionEngine()

    def successful_tool():
        return 42

    result = engine.execute(successful_tool)

    assert isinstance(result, CorrectionResult)
    assert result.success is True
    assert result.result == 42
    assert result.attempts == 1
    assert result.corrected is False


def test_failed_tool_retries():
    engine = SelfCorrectionEngine(max_retries=2)

    calls = {"count": 0}

    def failing_tool():
        calls["count"] += 1
        raise RuntimeError("temporary failure")

    result = engine.execute(failing_tool)

    assert result.success is False
    assert result.attempts == 3
    assert calls["count"] == 3
    assert result.error == "temporary failure"
    assert result.corrected is True


def test_retry_then_success():
    engine = SelfCorrectionEngine(max_retries=2)

    calls = {"count": 0}

    def unstable_tool():
        calls["count"] += 1

        if calls["count"] < 2:
            raise RuntimeError("temporary failure")

        return "success"

    result = engine.execute(unstable_tool)

    assert result.success is True
    assert result.result == "success"
    assert result.attempts == 2
    assert result.corrected is True


def test_zero_retry():
    engine = SelfCorrectionEngine(max_retries=0)

    calls = {"count": 0}

    def failing_tool():
        calls["count"] += 1
        raise RuntimeError("failure")

    result = engine.execute(failing_tool)

    assert result.success is False
    assert result.attempts == 1
    assert calls["count"] == 1


def test_negative_retry_rejected():
    with pytest.raises(ValueError):
        SelfCorrectionEngine(max_retries=-1)


def test_should_retry():
    engine = SelfCorrectionEngine(max_retries=2)

    assert engine.should_retry(0) is True
    assert engine.should_retry(1) is True
    assert engine.should_retry(2) is True
    assert engine.should_retry(3) is False


def test_correct_error():
    engine = SelfCorrectionEngine(max_retries=2)

    result = engine.correct_error(
        error="Invalid SQL",
        original_input="show revenue",
    )

    assert result["error"] == "Invalid SQL"
    assert result["original_input"] == "show revenue"
    assert result["action"] == "retry"
    assert result["attempt_limit"] == 2


def test_function_arguments_are_preserved():
    engine = SelfCorrectionEngine()

    def add(a, b):
        return a + b

    result = engine.execute(add, 10, 20)

    assert result.success is True
    assert result.result == 30
    assert result.attempts == 1


def test_keyword_arguments_are_preserved():
    engine = SelfCorrectionEngine()

    def multiply(a, b=1):
        return a * b

    result = engine.execute(
        multiply,
        a=5,
        b=4,
    )

    assert result.success is True
    assert result.result == 20