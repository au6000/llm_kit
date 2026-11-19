"""OpenAI — ツール呼び出しの引数を分割して受ける。

引数を**生の JSON 文字列の断片**として送ってくる。`langchain_openai` はその断片を
そのまま `tool_call_chunks[].args` に渡す（累積しない）ので、`ChatOpenAI` の astream に
そのまま乗れる。断片は JSON の途中なので、本文の引数の中身がどこまで確定したかだけ
こちらで数える。

    {"is_closing":false,"message":"お問い合わせ      ← ここまで届いた
                                   ^^^^^^^^^^ この分だけを本文の delta として出す
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from langchain_core.messages import BaseMessage

from ..types import ArgDelta, BodyArgResolver, StreamedResult, StreamedToolCall
from ..usage_tracker import add_usage

#: 1 回の呼び出しで待つ上限（秒）。`bounded=True` のときだけ使う。vertex と同じ値。
_REQUEST_TIMEOUT_SEC = 45

__all__ = ["BodyArgExtractor", "stream_tool_calls"]


# ── OpenAI ──────────────────────────────────────────────────────────────────


def _tool_name(tool: Any) -> str | None:
    """LangChain のツールでも、生の dict の宣言でも名前を引く。"""
    name = getattr(tool, "name", None)
    if name is None and isinstance(tool, dict):
        name = tool.get("name") or (tool.get("function") or {}).get("name")
    return name


# JSON のエスケープは最長 6 文字（\uXXXX）。境目で切れた分を出さないために保留する。
_HOLD_BACK = 6


class BodyArgExtractor:
    """引数の JSON の断片をつなぎ、指定した key の文字列が伸びたぶんを返す。

    **1 つのツール呼び出しにつき 1 個**持つ。呼び出しごとに JSON が別なので。
    """

    def __init__(self, key: str) -> None:
        self._needle = json.dumps(key, ensure_ascii=False) + ":"
        self._buf = ""
        self._start = -1  # 値の開き引用符の次の位置
        self._emitted = 0  # 値のうち、生の JSON として読み終えた長さ

    def feed(self, fragment: str) -> str:
        self._buf += fragment
        if self._start < 0:
            at = self._buf.find(self._needle)
            if at < 0:
                return ""
            quote = self._buf.find('"', at + len(self._needle))
            if quote < 0:
                return ""
            self._start = quote + 1
            self._emitted = 0

        raw = self._buf[self._start :]
        # 値の終わりは、エスケープされていない引用符
        end = -1
        i = 0
        while i < len(raw):
            char = raw[i]
            if char == "\\":
                i += 2
                continue
            if char == '"':
                end = i
                break
            i += 1
        available = end if end >= 0 else max(0, len(raw) - _HOLD_BACK)
        if available <= self._emitted:
            return ""
        chunk = raw[self._emitted : available]
        # 末尾がエスケープの途中なら、そこまで戻す
        while chunk and chunk.count("\\") % 2 == 1 and chunk.endswith("\\"):
            chunk = chunk[:-1]
        if not chunk:
            return ""
        self._emitted += len(chunk)
        try:
            return json.loads(f'"{chunk}"')
        except ValueError:
            # 途中で切れた分は次に回す
            self._emitted -= len(chunk)
            return ""


class _OpenAICalls:
    """`tool_call_chunks[].index` で呼び出しを振り分ける。

    OpenAI は並列呼び出しに index を振ってくるので、そこを見れば足りる。ただし index は
    0 から連番とは限らないため、こちら側の並び（call_index）に写して持つ。
    """

    def __init__(self, body_arg_for: BodyArgResolver | None) -> None:
        self.calls: list[StreamedToolCall] = []
        self.fragments: list[list[str]] = []
        self.extractors: list[BodyArgExtractor | None] = []
        self.body_args: list[str | None] = []
        self._by_wire: dict[Any, int] = {}
        self._body_arg_for = body_arg_for

    def resolve(self, wire_index: Any) -> int:
        key = wire_index if wire_index is not None else 0
        if key in self._by_wire:
            return self._by_wire[key]
        index = len(self.calls)
        self._by_wire[key] = index
        self.calls.append(StreamedToolCall(call_index=index))
        self.fragments.append([])
        self.extractors.append(None)
        self.body_args.append(None)
        return index

    def name_arrived(self, index: int, name: str) -> None:
        self.calls[index].name = name
        body_arg = self._body_arg_for(name) if self._body_arg_for else None
        self.body_args[index] = body_arg
        self.extractors[index] = BodyArgExtractor(body_arg) if body_arg else None


async def stream_tool_calls(
    messages: list[BaseMessage],
    tools: list[Any],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    effort: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    body_arg_for: BodyArgResolver | None = None,
    track_usage: bool = True,
    allowed_tool_names: list[str] | None = None,
    bounded: bool = True,
    location: str | None = None,
    priority: bool = False,
) -> AsyncIterator[tuple[str, Any]]:
    """`temperature`・`location`・`priority` は Vertex 用なのでここでは使わない（温度と上限は
    reasoning 系のモデルが弾くため、そもそも送らない）。
    振り分け（`providers.resolve_tool_call_streamer`）が同じ形で呼べるよう受けている。

    `allowed_tool_names` を渡すと、そのターンに呼べるツールをその名前だけに絞る。
    **vertex と絞り方が違う。** OpenAI には「この中から 1 つ」に当たる指定が無い
    （`tool_choice` は 1 つに固定するか任意かの二択）ので、宣言そのものを絞る。
    宣言が変わるとプロンプトのキャッシュの効きも変わる。
    """
    if not model:
        raise RuntimeError("model is required for the openai provider")

    if allowed_tool_names:
        allowed = set(allowed_tool_names)
        tools = [t for t in tools if _tool_name(t) in allowed]
        if not tools:
            raise RuntimeError(
                "allowed_tool_names のどれも渡されたツールに無い "
                f"(allowed={sorted(allowed)})"
            )

    extra: dict[str, Any] = {}
    effort = effort.strip() if effort else None
    if effort and effort != "none":
        # **比較のあいだだけ。**gpt-6-luna は /v1/chat/completions だと、ツールと
        # reasoning_effort（none 以外）を同時に受けない（400: "use /v1/responses or set
        # reasoning_effort to 'none'"）。none 以外のときだけ Responses API に切り替える。
        extra["use_responses_api"] = True
        extra["reasoning"] = {"effort": effort}
    elif effort:
        extra["reasoning_effort"] = effort
    if bounded:
        # 待ち時間に上限を付け、SDK の送り直し（既定 2 回）はさせない。止まったときや
        # 429 のときに、呼び出し側がすぐ別のモデルに切り替えられるようにする
        # （呼び出し側でモデルを切り替える仕組みと組み合わせる前提）。上限は vertex と同じ。
        extra["timeout"] = _REQUEST_TIMEOUT_SEC
        extra["max_retries"] = 0
    if api_key:
        extra["api_key"] = api_key
    if base_url:
        extra["base_url"] = base_url

    from ..llm import get_llm  # 遅延 import（llm.py との循環を避ける）

    llm = get_llm(
        "openai",
        model,
        # 旧実装と同じく温度と上限は送らない（reasoning 系のモデルが弾くため）。
        # None は ChatOpenAI 側の exclude_if_none で落ちる。
        temperature=None,
        max_tokens=None,
        # 使用量はこのモジュールが記録する（provider ごとに形を揃えるため）。
        track_usage=False,
        stream_usage=True,
        **extra,
    )
    bound = llm.bind_tools(tools, tool_choice="required")

    tracked = _OpenAICalls(body_arg_for)
    result = StreamedResult()
    usage: tuple[int, int] | None = None

    async for chunk in bound.astream(messages):
        meta = getattr(chunk, "usage_metadata", None)
        if meta:
            usage = (int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0))
        text = chunk.content
        if isinstance(text, str) and text:
            result.text += text
        for wire_call in chunk.tool_call_chunks or []:
            index = tracked.resolve(wire_call.get("index"))
            name = wire_call.get("name")
            if name and tracked.calls[index].name is None:
                tracked.name_arrived(index, name)
                yield "name", (index, name)
            fragment = wire_call.get("args")
            if not fragment:
                continue
            tracked.fragments[index].append(fragment)
            extractor = tracked.extractors[index]
            body_arg = tracked.body_args[index]
            if extractor is not None and body_arg:
                piece = extractor.feed(fragment)
                if piece:
                    yield "delta", ArgDelta(index, body_arg, None, piece)

    for call, fragments in zip(tracked.calls, tracked.fragments):
        joined = "".join(fragments)
        if not joined:
            continue
        try:
            call.args = json.loads(joined)
        except ValueError as e:
            # 引数の中身は利用者の発言を引くことがあるので、メッセージには入れない
            # （この例外は呼び出し側で ERROR のログに出る）。原因の位置は `from e` の
            # JSONDecodeError（位置だけで中身は含まない）で分かる。
            raise RuntimeError(
                f"OpenAI のツール引数が JSON として読めない (call {call.call_index}, "
                f"tool={call.name}, len={len(joined)})"
            ) from e
    result.calls = tracked.calls
    if track_usage and usage:
        add_usage("openai", model, usage[0], usage[1])
    yield "done", result
