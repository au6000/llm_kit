"""
llm_kit - LLM を呼ぶときに要る機能をひととおり揃えたライブラリ（中身は LangChain の上に作っている）

使い方、入れ方、プロバイダごとの環境変数は llm_kit/README.md。

    from llm_kit import get_llm, invoke, print_usage_summary

    llm = get_llm("vertex", "gemini-3-flash-preview")  # LangChain の ChatModel
    print(llm.invoke("Hello").text)
    print(invoke("Hello", provider="openai", model="gpt-6.2-sol"))
    print_usage_summary()
"""

from importlib import import_module

# 追加の依存が要らない部品（標準ライブラリだけで書いてある）は、ここで普通に読む。
from .config import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    EMBEDDING_PRICING,
    PRICING,
)
from .exceptions import (
    LLMAPIError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from .rerank import rerank
from .types import ArgDelta, BodyArgResolver, StreamedResult, StreamedToolCall, UsageRecord

# LangChain が要る部品は、名前が初めて使われた時点で読む（PEP 562 のモジュールの __getattr__）。
# こうしておくと、llm_kit の依存を入れていない環境（たとえば numpy だけの prompt-lab の本体）でも
# `import llm_kit` が通り、リポジトリ直下の `python -m unittest` がここで落ちない。
_LAZY = {
    "llm": (
        "get_llm",
        "get_structured_llm",
        "invoke",
        "invoke_structured",
        "ainvoke",
        "ainvoke_structured",
        "ainvoke_structured_with_retry",
        "stream",
        "astream",
        "astream_tool_calls",
        "configure_llm_concurrency",
        "get_llm_semaphore",
    ),
    "providers.vertex": ("thought_signature_kwargs",),
    "providers.claude_code": ("ChatClaudeCode",),
    "embedding": ("get_embeddings", "embed_query", "embed_documents", "aembed_query", "aembed_documents"),
    "utils": ("try_lenient_parse", "build_retry_messages"),
    "usage_tracker": (
        "print_usage_summary",
        "get_usage_stats",
        "get_usage_details",
        "reset_usage_stats",
        "save_usage",
        "estimate_cost",
        "estimate_batch_cost",
        "print_estimate",
    ),
}
_MODULE_OF = {name: module for module, names in _LAZY.items() for name in names}
_MODULE_OF["tracker"] = "usage_tracker"  # 部品 usage_tracker そのものの別名

__all__ = [
    "PRICING",
    "DEFAULT_MODELS",
    "DEFAULT_PROVIDER",
    "DEFAULT_EMBEDDING_MODELS",
    "EMBEDDING_PRICING",
    "LLMError",
    "LLMConfigError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMResponseError",
    "rerank",
    "ArgDelta",
    "BodyArgResolver",
    "StreamedResult",
    "StreamedToolCall",
    "UsageRecord",
    *_MODULE_OF,
]

__version__ = "2.0.0"


def __getattr__(name: str):
    module_name = _MODULE_OF.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = module if name == "tracker" else getattr(module, name)
    globals()[name] = value  # 2 回目からは普通の属性として引く
    return value

# __dir__ は足さない。unittest の探索は dir() の名前を全部引くので、遅らせた名前を dir() に
# 見せると、そこで LangChain を読みに行ってしまう。
