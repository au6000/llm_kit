"""llm_kit が外に見せる型。

**データの形だけを置く。** 振る舞いを持つもの（`UsageTracker`、`ChatClaudeCode`、
引数を組み立てる器など）は実装のあるファイルに残す。例外は `exceptions.py`。

ここに集めているのは、呼び出し側が「返ってくるものの形」だけを知りたいときに
実装を読まなくて済むようにするため。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

__all__ = [
    "ArgDelta",
    "BodyArgResolver",
    "StreamedResult",
    "StreamedToolCall",
    "UsageRecord",
]


# ── ツール呼び出しを流して受ける（llm.astream_tool_calls） ──────────────────

BodyArgResolver = Callable[[str], "str | None"]
"""ツール名 -> 顧客向けの本文が入る引数の名前（無ければ None）。

どの引数が本文かはドメインの取り決めなので、この層では持たず呼び出し側から渡す。
"""


@dataclass
class ArgDelta:
    """文字列の引数が増えたぶん。"""

    call_index: int
    """何番目のツール呼び出しか。"""
    key: str
    """引数の名前。"""
    item_index: int | None
    """配列の要素ならその添字。スカラーなら None。"""
    text: str


@dataclass
class StreamedToolCall:
    """流し終わったあとの、1 つのツール呼び出し。"""

    call_index: int
    name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    # 次のターンの履歴に載せるために base64 で持つ（LangChain と同じ持ち方）。
    thought_signature: str | None = None


@dataclass
class StreamedResult:
    """1 回のモデル呼び出しの結果。

    `calls` は 1 ターンに複数入りうる（並列ツール呼び出し）。`tool_choice="required"` も
    `mode="ANY"` も「1 回以上」の指定で、回数を 1 に固定しない。
    """

    calls: list[StreamedToolCall] = field(default_factory=list)
    # ツール呼び出しを強制しているので通常は空。出たときに黙って捨てないため残す。
    text: str = ""

    @property
    def first(self) -> StreamedToolCall | None:
        """1 ターンに 1 つしか呼ばせない呼び出し側のための近道。"""
        return self.calls[0] if self.calls else None


# ── 使用量の記録（usage_tracker） ────────────────────────────────────────────


@dataclass
class UsageRecord:
    """1 回の API 呼び出しの記録。"""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_time: float  # 秒
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 出力（output_tokens）のうち、モデルが考えるのに使ったトークン。分からなければ None。
    thinking_tokens: int | None = None
    # 依頼を送ってから、応答の最初のかたまりが届くまでの秒数。測っていなければ None。
    first_chunk_seconds: float | None = None
