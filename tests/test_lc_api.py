from app.core.config import settings

topic = "Tell me a joke"


def test_langchain_google_genai_chat():
    from langchain_core.messages import AIMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", api_key=settings.GOOGLE_API_KEY
    )

    resp = llm.invoke(topic)

    assert isinstance(resp, AIMessage)
    assert isinstance(resp.content, str)


def test_langchain_hf_endpoint():
    from langchain_huggingface import HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        model="microsoft/Phi-3-mini-4k-instruct",
    )

    topic = "Tell me a joke"
    resp = llm.invoke(topic)
    assert isinstance(resp, str)
