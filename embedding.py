"""
テキストエンベディング取得
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_google_vertexai")

from langchain_core.embeddings import Embeddings

from .config import DEFAULT_PROVIDER, DEFAULT_EMBEDDING_MODELS
from .exceptions import LLMConfigError


def get_embeddings(
    provider: str = None,
    model: str = None,
    **kwargs,
) -> Embeddings:
    """
    LangChain Embeddingsを取得

    Args:
        provider: プロバイダー名 (openai, google, vertex)
        model: モデル名
        **kwargs: 追加パラメータ

    Returns:
        Embeddings: LangChain Embeddings

    Example:
        # 基本
        embeddings = get_embeddings("vertex")
        vector = embeddings.embed_query("Hello")
        print(len(vector))  # 768

        # 複数テキストの埋め込み
        vectors = embeddings.embed_documents(["Hello", "World"])
        print(len(vectors))  # 2
    """
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_EMBEDDING_MODELS.get(provider)

    if not model:
        raise LLMConfigError(f"No default embedding model for provider: {provider}")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            **kwargs,
        )

    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from .utils import get_google_credentials

        credentials, project_id = get_google_credentials()

        return GoogleGenerativeAIEmbeddings(
            model=model,
            credentials=credentials,
            **kwargs,
        )

    elif provider == "vertex":
        from langchain_google_vertexai import VertexAIEmbeddings
        from .utils import get_vertex_config

        config = get_vertex_config()

        return VertexAIEmbeddings(
            model_name=model,
            project=config["project_id"],
            location=config["location"],
            **kwargs,
        )

    else:
        raise LLMConfigError(f"Unknown provider for embeddings: {provider}")


def embed_query(
    text: str,
    provider: str = None,
    model: str = None,
) -> list[float]:
    """
    単一テキストの埋め込みベクトルを取得

    Args:
        text: 埋め込み対象のテキスト
        provider: プロバイダー名
        model: モデル名

    Returns:
        list[float]: 埋め込みベクトル

    Example:
        vector = embed_query("Hello world")
        print(len(vector))  # 768 or 1536 depending on model
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return embeddings.embed_query(text)


def embed_documents(
    texts: list[str],
    provider: str = None,
    model: str = None,
) -> list[list[float]]:
    """
    複数テキストの埋め込みベクトルを取得

    Args:
        texts: 埋め込み対象のテキストリスト
        provider: プロバイダー名
        model: モデル名

    Returns:
        list[list[float]]: 埋め込みベクトルのリスト

    Example:
        vectors = embed_documents(["Hello", "World"])
        print(len(vectors))  # 2
        print(len(vectors[0]))  # 768 or 1536 depending on model
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return embeddings.embed_documents(texts)


async def aembed_query(
    text: str,
    provider: str = None,
    model: str = None,
) -> list[float]:
    """
    単一テキストの埋め込みベクトルを取得（非同期版）

    Args:
        text: 埋め込み対象のテキスト
        provider: プロバイダー名
        model: モデル名

    Returns:
        list[float]: 埋め込みベクトル
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return await embeddings.aembed_query(text)


async def aembed_documents(
    texts: list[str],
    provider: str = None,
    model: str = None,
) -> list[list[float]]:
    """
    複数テキストの埋め込みベクトルを取得（非同期版）

    Args:
        texts: 埋め込み対象のテキストリスト
        provider: プロバイダー名
        model: モデル名

    Returns:
        list[list[float]]: 埋め込みベクトルのリスト
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return await embeddings.aembed_documents(texts)
