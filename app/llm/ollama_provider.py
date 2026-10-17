from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class OllamaProvider:
    """
    LangChain-based local LLM provider.

    The provider is intentionally isolated from the rest of the
    application so the underlying model can be replaced later.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Send a prompt to the local LLM.
        """

        messages = []

        if system_prompt:
            messages.append(
                SystemMessage(content=system_prompt)
            )

        messages.append(
            HumanMessage(content=prompt)
        )

        response = self.llm.invoke(messages)

        return response.content

    def structured_invoke(
        self,
        schema: Any,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Any:
        """
        Invoke the LLM with a structured-output schema.
        """

        structured_llm = self.llm.with_structured_output(schema)

        messages = []

        if system_prompt:
            messages.append(
                SystemMessage(content=system_prompt)
            )

        messages.append(
            HumanMessage(content=prompt)
        )

        return structured_llm.invoke(messages)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }