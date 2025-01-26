"""
Provide several LLM APIs:
1. Gemini.
2. HuggingFace.
3. Groq.
"""


class LLMsProvider:
    @staticmethod
    def build_model(provider: str, model: str, api_key: str, **kwargs):
        """
        Build and return an LLM instance based on the provider.

        Args:
            provider (str): The LLM provider ('gemini', 'huggingface', 'groq').
            model (str): The model name or identifier.
            api_key (str): The API key for authentication.
            **kwargs: Additional arguments for the LLM initialization.

        Returns:
            An instance of the LLM.

        Raises:
            ImportError: If the required package for the provider is not installed.
            ValueError: If the provider name is invalid.
        """
        if not provider:
            raise ValueError("Provider name must be specified.")

        try:
            match provider.lower():
                case "gemini":
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                    except ImportError as e:
                        raise ImportError(
                            "The package 'langchain_google_genai' is not installed."
                        ) from e

                    return ChatGoogleGenerativeAI(
                        model=model, api_key=api_key, **kwargs
                    )

                case "groq":
                    try:
                        from langchain_groq import ChatGroq
                    except ImportError as e:
                        raise ImportError(
                            "The package 'langchain_groq' is not installed."
                        ) from e

                    return ChatGroq(model=model, api_key=api_key, **kwargs)

                case "huggingface":
                    try:
                        from langchain_huggingface import (
                            ChatHuggingFace,
                            HuggingFaceEndpoint,
                        )
                    except ImportError as e:
                        raise ImportError(
                            "The package 'langchain_huggingface' is not installed."
                        ) from e

                    endpoint = HuggingFaceEndpoint(
                        model=model, huggingfacehub_api_token=api_key, **kwargs
                    )
                    return ChatHuggingFace(endpoint, verbose=False)

                case _:
                    raise ValueError(
                        f"Unsupported provider: '{provider}'. Available providers are 'gemini', 'huggingface', and 'groq'."
                    )

        except Exception as e:
            # Log or re-raise the error with additional context
            raise RuntimeError(
                f"Failed to initialize LLM for provider '{provider}': {e}"
            ) from e
