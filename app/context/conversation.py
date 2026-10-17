from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ConversationMessage:
    role: str
    content: str
    timestamp: str


class ConversationContext:
    """
    Session-scoped conversation context.

    Stores recent user/assistant messages and optional analytical
    context such as the active dataset, previous results, and evidence.

    This class is intentionally storage-agnostic.
    Redis will be connected in the next step.
    """

    VALID_ROLES = {"user", "assistant", "system"}

    def __init__(
        self,
        session_id: str,
        max_messages: int = 20,
    ):
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")

        session_id = session_id.strip()

        if not session_id:
            raise ValueError("session_id cannot be empty")

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than 0"
            )

        self.session_id = session_id
        self.max_messages = max_messages

        self.messages: list[ConversationMessage] = []

        self.dataset_id: str | None = None
        self.dataset_version: str | None = None
        self.last_result: Any = None
        self.last_evidence: list[dict[str, Any]] = []

    def add_message(
        self,
        role: str,
        content: str,
    ) -> ConversationMessage:
        """
        Add a message to the current session.
        """

        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role: {role}"
            )

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string"
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "content cannot be empty"
            )

        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        self.messages.append(message)

        # Keep only the most recent messages.
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[
                -self.max_messages:
            ]

        return message

    def add_user_message(
        self,
        content: str,
    ) -> ConversationMessage:
        return self.add_message(
            role="user",
            content=content,
        )

    def add_assistant_message(
        self,
        content: str,
    ) -> ConversationMessage:
        return self.add_message(
            role="assistant",
            content=content,
        )

    def add_system_message(
        self,
        content: str,
    ) -> ConversationMessage:
        return self.add_message(
            role="system",
            content=content,
        )

    def get_messages(self) -> list[ConversationMessage]:
        return list(self.messages)

    def get_recent_messages(
        self,
        limit: int = 5,
    ) -> list[ConversationMessage]:
        if limit <= 0:
            return []

        return self.messages[-limit:]

    def get_history_text(
        self,
        limit: int | None = None,
    ) -> str:
        """
        Convert conversation history into a compact text representation.
        """

        messages = (
            self.messages
            if limit is None
            else self.get_recent_messages(limit)
        )

        return "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

    def set_dataset_context(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
    ) -> None:
        if not isinstance(dataset_id, str):
            raise TypeError(
                "dataset_id must be a string"
            )

        dataset_id = dataset_id.strip()

        if not dataset_id:
            raise ValueError(
                "dataset_id cannot be empty"
            )

        self.dataset_id = dataset_id
        self.dataset_version = dataset_version

    def set_result(
        self,
        result: Any,
    ) -> None:
        self.last_result = result

    def set_evidence(
        self,
        evidence: list[dict[str, Any]],
    ) -> None:
        if not isinstance(evidence, list):
            raise TypeError(
                "evidence must be a list"
            )

        self.last_evidence = evidence

    def get_context(self) -> dict[str, Any]:
        """
        Return the complete session context.
        """

        return {
            "session_id": self.session_id,
            "messages": [
                asdict(message)
                for message in self.messages
            ],
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "last_result": self.last_result,
            "last_evidence": self.last_evidence,
        }

    def clear(self) -> None:
        """
        Clear the session context.
        """

        self.messages.clear()
        self.dataset_id = None
        self.dataset_version = None
        self.last_result = None
        self.last_evidence = []

    def message_count(self) -> int:
        return len(self.messages)