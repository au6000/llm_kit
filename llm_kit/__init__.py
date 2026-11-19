"""
llm_kit - LLM を呼ぶときに要る機能をひととおり揃えたライブラリ（中身は LangChain の上に作っている）

Usage:
    from llm_kit import get_llm, print_usage_summary

    # 基本
    llm = get_llm("vertex", "gemini-2.0-flash")
    response = llm.invoke("Hello")
    print(response.content)

    # Structured Output（型安全なJSON）
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str
        confidence: float

    structured_llm = get_llm("openai", "gpt-4o").with_structured_output(Answer)
    result = structured_llm.invoke("What is 2+2?")
    print(result.answer)  # "4"

    # Streaming
    for chunk in llm.stream("Tell me a story"):
        print(chunk.content, end="", flush=True)

    # 使用量確認
    print_usage_summary()

    # LangGraph連携
    def call_model(state):
        llm = get_llm("anthropic", "claude-sonnet-4-20250514")
        return {"messages": [llm.invoke(state["messages"])]}

Required packages:
    使うプロバイダの分だけ追加で入れる（pyproject.toml の optional-dependencies）。
    全部なら: uv add "llm-kit[all] @ git+https://github.com/au6000/llm_kit.git"

Required environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
    - VERTEX_PROJECT_ID (for Vertex AI)
"""

from .llm import (
    get_llm,
    get_structured_llm,
    invoke,
    invoke_structured,
    ainvoke,
    ainvoke_structured,
    ainvoke_structured_with_retry,
    stream,
    astream_tool_calls,
    configure_llm_concurrency,
    get_llm_semaphore,
)

from .providers.vertex import thought_signature_kwargs

from .types import (
    ArgDelta,
    BodyArgResolver,
    StreamedResult,
    StreamedToolCall,
    UsageRecord,
)

from .providers.claude_code import ChatClaudeCode

from .embedding import (
    get_embeddings,
    embed_query,
    embed_documents,
    aembed_query,
    aembed_documents,
)

from .rerank import rerank

from .exceptions import (
    LLMError,
    LLMConfigError,
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseError,
)

from .utils import (
    try_lenient_parse,
    build_retry_messages,
)

from . import usage_tracker as tracker
from .usage_tracker import (
    print_usage_summary,
    get_usage_stats,
    get_usage_details,
    reset_usage_stats,
    save_usage,
    estimate_cost,
    estimate_batch_cost,
    print_estimate,
)

from .config import (
    PRICING,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    DEFAULT_EMBEDDING_MODELS,
    EMBEDDING_PRICING,
)

from .query_backend import (
    QueryBackend,
    SqliteBackend,
    make_backend,
)

__all__ = [
    # Main
    "get_llm",
    "astream_tool_calls",
    "ArgDelta",
    "BodyArgResolver",
    "StreamedResult",
    "StreamedToolCall",
    "UsageRecord",
    "thought_signature_kwargs",
    "get_structured_llm",
    "invoke",
    "invoke_structured",
    "ainvoke",
    "ainvoke_structured",
    "ainvoke_structured_with_retry",
    "stream",
    "configure_llm_concurrency",
    "get_llm_semaphore",
    "ChatClaudeCode",
    # Embeddings
    "get_embeddings",
    "embed_query",
    "embed_documents",
    "aembed_query",
    "aembed_documents",
    # Rerank
    "rerank",
    # Query backend
    "QueryBackend",
    "SqliteBackend",
    "make_backend",
    # Exceptions
    "LLMError",
    "LLMConfigError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMResponseError",
    # Validation
    "try_lenient_parse",
    "build_retry_messages",
    # Tracker
    "tracker",
    "print_usage_summary",
    "get_usage_stats",
    "get_usage_details",
    "reset_usage_stats",
    "save_usage",
    "estimate_cost",
    "estimate_batch_cost",
    "print_estimate",
    # Config
    "PRICING",
    "DEFAULT_MODELS",
    "DEFAULT_PROVIDER",
    "DEFAULT_EMBEDDING_MODELS",
    "EMBEDDING_PRICING",
]

__version__ = "1.0.0"
