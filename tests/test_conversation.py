import pytest

from app.context.conversation import (
    ConversationContext,
)


def test_context_initialization():
    context = ConversationContext(
        session_id="session-001"
    )

    assert context.session_id == "session-001"
    assert context.message_count() == 0


def test_empty_session_id():
    with pytest.raises(ValueError):
        ConversationContext("")


def test_whitespace_session_id():
    with pytest.raises(ValueError):
        ConversationContext("   ")


def test_invalid_session_id_type():
    with pytest.raises(TypeError):
        ConversationContext(None)


def test_add_user_message():
    context = ConversationContext("session-001")

    message = context.add_user_message(
        "What is the total revenue?"
    )

    assert message.role == "user"
    assert message.content == (
        "What is the total revenue?"
    )
    assert context.message_count() == 1


def test_add_assistant_message():
    context = ConversationContext("session-001")

    message = context.add_assistant_message(
        "The total revenue is 6000."
    )

    assert message.role == "assistant"
    assert message.content == (
        "The total revenue is 6000."
    )


def test_add_system_message():
    context = ConversationContext("session-001")

    message = context.add_system_message(
        "Use deterministic tools for calculations."
    )

    assert message.role == "system"


def test_invalid_role():
    context = ConversationContext("session-001")

    with pytest.raises(ValueError):
        context.add_message(
            "invalid",
            "test",
        )


def test_empty_content():
    context = ConversationContext("session-001")

    with pytest.raises(ValueError):
        context.add_message(
            "user",
            "",
        )


def test_whitespace_content():
    context = ConversationContext("session-001")

    with pytest.raises(ValueError):
        context.add_message(
            "user",
            "   ",
        )


def test_non_string_content():
    context = ConversationContext("session-001")

    with pytest.raises(TypeError):
        context.add_message(
            "user",
            None,
        )


def test_message_limit():
    context = ConversationContext(
        session_id="session-001",
        max_messages=3,
    )

    context.add_user_message("message 1")
    context.add_user_message("message 2")
    context.add_user_message("message 3")
    context.add_user_message("message 4")

    assert context.message_count() == 3

    messages = context.get_messages()

    assert messages[0].content == "message 2"
    assert messages[-1].content == "message 4"


def test_get_recent_messages():
    context = ConversationContext("session-001")

    for index in range(5):
        context.add_user_message(
            f"message {index}"
        )

    recent = context.get_recent_messages(2)

    assert len(recent) == 2
    assert recent[0].content == "message 3"
    assert recent[1].content == "message 4"


def test_recent_limit_zero():
    context = ConversationContext("session-001")

    context.add_user_message("message")

    assert context.get_recent_messages(0) == []


def test_history_text():
    context = ConversationContext("session-001")

    context.add_user_message(
        "What is revenue?"
    )

    context.add_assistant_message(
        "Revenue is 6000."
    )

    history = context.get_history_text()

    assert "user: What is revenue?" in history
    assert "assistant: Revenue is 6000." in history


def test_history_text_with_limit():
    context = ConversationContext("session-001")

    context.add_user_message("first")
    context.add_user_message("second")
    context.add_user_message("third")

    history = context.get_history_text(2)

    assert "first" not in history
    assert "second" in history
    assert "third" in history


def test_dataset_context():
    context = ConversationContext("session-001")

    context.set_dataset_context(
        dataset_id="sales_123",
        dataset_version="v1",
    )

    assert context.dataset_id == "sales_123"
    assert context.dataset_version == "v1"


def test_invalid_dataset_id():
    context = ConversationContext("session-001")

    with pytest.raises(ValueError):
        context.set_dataset_context("")


def test_dataset_id_type():
    context = ConversationContext("session-001")

    with pytest.raises(TypeError):
        context.set_dataset_context(None)


def test_set_result():
    context = ConversationContext("session-001")

    result = {
        "revenue": 6000,
        "profit": 2400,
    }

    context.set_result(result)

    assert context.last_result == result


def test_set_evidence():
    context = ConversationContext("session-001")

    evidence = [
        {
            "tool": "sql_query",
            "result": 6000,
        }
    ]

    context.set_evidence(evidence)

    assert context.last_evidence == evidence


def test_invalid_evidence():
    context = ConversationContext("session-001")

    with pytest.raises(TypeError):
        context.set_evidence(None)


def test_get_context():
    context = ConversationContext("session-001")

    context.add_user_message(
        "What is revenue?"
    )

    context.set_dataset_context(
        "sales_123",
        "v1",
    )

    context.set_result(
        {"revenue": 6000}
    )

    context.set_evidence(
        [{"tool": "sql_query"}]
    )

    result = context.get_context()

    assert result["session_id"] == "session-001"
    assert len(result["messages"]) == 1
    assert result["dataset_id"] == "sales_123"
    assert result["dataset_version"] == "v1"
    assert result["last_result"]["revenue"] == 6000
    assert len(result["last_evidence"]) == 1


def test_clear():
    context = ConversationContext("session-001")

    context.add_user_message("Question")
    context.set_dataset_context("sales")
    context.set_result({"value": 10})
    context.set_evidence([{"tool": "statistics"}])

    context.clear()

    assert context.message_count() == 0
    assert context.dataset_id is None
    assert context.dataset_version is None
    assert context.last_result is None
    assert context.last_evidence == []


def test_multiple_turn_conversation():
    context = ConversationContext("session-001")

    context.add_user_message(
        "What is total revenue?"
    )

    context.add_assistant_message(
        "Total revenue is 6000."
    )

    context.add_user_message(
        "Which region contributed most?"
    )

    context.add_assistant_message(
        "South contributed most."
    )

    assert context.message_count() == 4

    history = context.get_history_text()

    assert "total revenue" in history
    assert "South" in history