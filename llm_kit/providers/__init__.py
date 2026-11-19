"""provider ごとの実装。

ここに置くのは「その provider でしか意味を持たないもの」だけ。呼び出しの入口は
`llm.py` にあり、provider 名でここに振り分ける。

今あるのは、ツール呼び出しの引数を流して受ける実装（vertex / openai）と、
claude_code の ChatModel 実装。`get_llm` の chat model 構築の分岐や、埋め込み・
rerank の分岐は、まだ `llm.py` / `embedding.py` / `rerank.py` に残っている。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

__all__ = ["resolve_tool_call_streamer"]

ToolCallStreamer = Callable[..., "AsyncIterator[tuple[str, Any]]"]


def resolve_tool_call_streamer(provider: str | None) -> ToolCallStreamer:
    """provider 名から、ツール呼び出しを流す実装を引く。

    既定は vertex。名前が増えたらここに 1 行足す。
    """
    if (provider or "").strip().lower() == "openai":
        from .openai import stream_tool_calls

        return stream_tool_calls
    from .vertex import stream_tool_calls

    return stream_tool_calls
