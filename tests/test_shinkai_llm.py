from unittest.mock import patch

import pytest

from app.core.config import settings
from app.modules.llms import LLMsProvider


@pytest.mark.parametrize(
    "provider,model,api_key",
    [
        ("gemini", "gemini-1.5-flash", settings.GOOGLE_API_KEY),
        ("groq", "mixtral-8x7b-32768", settings.GROQ_API_KEY),
        (
            "huggingface",
            "microsoft/Phi-3-mini-4k-instruct",
            settings.HUGGINGFACE_API_KEY,
        ),
    ],
)
def test_build_model_success(provider, model, api_key):
    """Test successful initialization of LLMs for all supported providers."""
    with (
        patch(f"langchain_google_genai.ChatGoogleGenerativeAI") as mock_gemini,
        patch(f"langchain_groq.ChatGroq") as mock_groq,
        patch(f"langchain_huggingface.ChatHuggingFace") as mock_hf,
        patch(f"langchain_huggingface.HuggingFaceEndpoint") as mock_hf_endpoint,
    ):
        if provider == "gemini":
            LLMsProvider.build_model(provider, model, api_key)
            mock_gemini.assert_called_once_with(model=model, api_key=api_key)

        elif provider == "groq":
            LLMsProvider.build_model(provider, model, api_key)
            mock_groq.assert_called_once_with(model=model, api_key=api_key)

        elif provider == "huggingface":
            LLMsProvider.build_model(provider, model, api_key)
            mock_hf_endpoint.assert_called_once_with(
                model=model, huggingfacehub_api_token=api_key
            )
            mock_hf.assert_called_once()


def test_invalid_provider():
    """Test invalid provider name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMsProvider.build_model(
            "openai", "microsoft/Phi-3-mini-4k-instruct", settings.HUGGINGFACE_API_KEY
        )


def test_missing_import_gemini():
    """Test missing import error for Gemini."""
    with patch(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        side_effect=ImportError("Module not found"),
    ):
        with pytest.raises(
            ImportError, match="The package 'langchain_google_genai' is not installed."
        ):
            LLMsProvider.build_model(
                "gemini", "gemini-1.5-flash", settings.GOOGLE_API_KEY
            )


def test_missing_import_groq():
    """Test missing import error for Groq."""
    with patch("langchain_groq.ChatGroq", side_effect=ImportError("Module not found")):
        with pytest.raises(
            ImportError, match="The package 'langchain_groq' is not installed."
        ):
            LLMsProvider.build_model(
                "groq", "mixtral-8x7b-32768", settings.GROQ_API_KEY
            )


def test_missing_import_huggingface():
    """Test missing import error for HuggingFace."""
    with patch(
        "langchain_huggingface.ChatHuggingFace",
        side_effect=ImportError("Module not found"),
    ):
        with pytest.raises(
            ImportError, match="The package 'langchain_huggingface' is not installed."
        ):
            LLMsProvider.build_model(
                "huggingface",
                "microsoft/Phi-3-mini-4k-instruct",
                settings.HUGGINGFACE_API_KEY,
            )


def test_empty_provider():
    """Test empty provider name raises ValueError."""
    with pytest.raises(ValueError, match="Provider name must be specified."):
        LLMsProvider.build_model(
            "", "microsoft/Phi-3-mini-4k-instruct", settings.HUGGINGFACE_API_KEY
        )


def test_exception_handling():
    """Test general exception handling for unexpected errors."""
    with patch(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        side_effect=Exception("Unexpected error"),
    ):
        with pytest.raises(
            RuntimeError, match="Failed to initialize LLM for provider 'gemini'"
        ):
            LLMsProvider.build_model(
                "gemini", "gemini-1.5-flash", settings.GOOGLE_API_KEY
            )
