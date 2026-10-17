from unittest.mock import MagicMock, patch

from app.llm.ollama_provider import OllamaProvider


def test_provider_initialization():
    with patch(
        "app.llm.ollama_provider.ChatOllama"
    ) as mock_llm:

        provider = OllamaProvider(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            temperature=0.0,
        )

        assert provider.model_name == "qwen2.5:3b"
        assert provider.base_url == "http://localhost:11434"
        assert provider.temperature == 0.0

        mock_llm.assert_called_once()


def test_get_model_info():
    with patch(
        "app.llm.ollama_provider.ChatOllama"
    ):

        provider = OllamaProvider()

        info = provider.get_model_info()

        assert info["provider"] == "ollama"
        assert info["model"] == "qwen2.5:3b"
        assert info["base_url"] == "http://localhost:11434"
        assert info["temperature"] == 0.0


def test_invoke():
    with patch(
        "app.llm.ollama_provider.ChatOllama"
    ) as mock_llm:

        mock_instance = mock_llm.return_value

        mock_response = MagicMock()
        mock_response.content = "Revenue increased by 15%."

        mock_instance.invoke.return_value = mock_response

        provider = OllamaProvider()

        result = provider.invoke(
            prompt="What happened to revenue?"
        )

        assert result == "Revenue increased by 15%."

        mock_instance.invoke.assert_called_once()


def test_invoke_with_system_prompt():
    with patch(
        "app.llm.ollama_provider.ChatOllama"
    ) as mock_llm:

        mock_instance = mock_llm.return_value

        mock_response = MagicMock()
        mock_response.content = "Analyze the data."

        mock_instance.invoke.return_value = mock_response

        provider = OllamaProvider()

        result = provider.invoke(
            prompt="Analyze revenue.",
            system_prompt="You are a BI analyst.",
        )

        assert result == "Analyze the data."

        messages = mock_instance.invoke.call_args.args[0]

        assert len(messages) == 2


def test_model_info_does_not_call_llm():
    with patch(
        "app.llm.ollama_provider.ChatOllama"
    ) as mock_llm:

        provider = OllamaProvider()

        provider.get_model_info()

        mock_llm.return_value.invoke.assert_not_called()