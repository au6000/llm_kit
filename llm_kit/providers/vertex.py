"""Vertex (Gemini) — ツール呼び出しの引数を分割して受ける。

`tool_config.function_calling_config.stream_function_call_arguments` を立てると
`FunctionCall.partial_args` に断片が、`will_continue` に続きの有無が乗る（Gemini 3 以降）。
断片は「jsonPath + 型付きの値」で来るので、ここで組み立てて引数の dict に戻す。

**`langchain-google-genai` は `partial_args` を解釈しない。** `tool_call_chunks` の args が
`{}` のまま積まれ、本文が丸ごと消える（4.3.5 時点。langchain-ai/langchain-google#1752）。
そのため `ChatGoogleGenerativeAI` の astream には乗せず、リクエストの組み立てだけ借りて
（`_prepare_request`）、チャンクの解釈をここで持つ。フラグ自体は `tool_config` を直接
渡せば通るので、履歴の変換・ツール宣言の変換・thought_signature の受け渡しは LangChain の
ものがそのまま効く。**upstream が対応したらこのファイルは openai.py と同じ形（astream に
乗るだけ）に畳める。**
"""

from __future__ import annotations

import base64
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, AsyncIterator

import httpx
from langchain_core.messages import BaseMessage

from ..types import ArgDelta, BodyArgResolver, StreamedResult, StreamedToolCall
from ..usage_tracker import add_usage

__all__ = ["ArgAssembler", "stream_tool_calls", "thought_signature_kwargs"]

logger = logging.getLogger(__name__)


# **待ち時間の上限を自分で決める。**`ChatGoogleGenerativeAI` は既定が
# `max_retries=6` / `timeout=None` で、429 が続くと指数バックオフを 6 回待ち切ってから
# 例外になる（実測で 1 呼び出し 600 秒）。本文を流し始めた後に止まると、利用者の画面は
# その間ずっと固まる。
#
# ストリーミングは送り始めたあとに握り直せないので、SDK のリトライは付けず、
# タイムアウトだけを必ず渡す。
#
# `attempts` は初回を含む試行回数なので 1 = 再送しない（SDK に再送させない）。
#
# **最初のかたまりが届く前の時間切れ・接続の失敗だけは、こちらで 1 回送り直す**
# （`_open_stream`）。応答のヘッダーすら返らないまま止まる呼び出しがときどきある
# （gemini-3.7-flash で出力の長いターンを何度も続けると、3 回に 3 回、どこかのターンで起きた）。
# まだ利用者に何も流していないので、送り直しても同じ文は 2 回出ない。
# 流し始めたあとの失敗は、上の理由どおり送り直さない。
#
# 1 回あたりの上限は 45 秒にする。普段は 2 秒前後だが、出力が長いターンでは
# ヘッダーまで 22 秒かかった実測がある（30 秒では余裕が無い）。送り直しを含めた上限は、
# 時間切れが 2 回続いたときの約 90 秒。普段の速さは変わらず、止まったときに諦めるまでの
# 時間だけが延びる。
#
# **1 ターンに時間がかかる前提の呼び出しには上限をかけない**（`bounded=False`）。
# 遅いだけの応答を打ち切ると、かえって失敗する（応答 36 秒のあと、次の呼び出しが
# 45 秒の打ち切りを 2 回続けて受けて失敗した）。
_REQUEST_TIMEOUT_SEC = 45
_REQUEST_ATTEMPTS = 1
_FIRST_CHUNK_ATTEMPTS = 2

# **出力の上限。**入れないと、モデルが生成を終えられない入力にあたったとき、
# 時間いっぱい書き続ける（実測で 60 秒・1,100 チャンク）。
#
# **上限は回答本文 2,000 字。**API はトークンでしか指定できないので、2,000 字が
# 確実に収まる値にする。日本語は 1 字あたり 0.42 トークンだったが（`count_tokens` で
# 実測）、比率は中身で動くので、1 字 1 トークンの最悪側で見ても足りる 2,048 を置く。
#
# 128 問の実測では、回答本文は中央値 630 字・最長 1,585 字で、2,000 字を超えたものは
# 1 件も無い。
_MAX_OUTPUT_TOKENS = 2048


#: 送り直してよい失敗。どれも、応答が 1 バイトも届く前に起きうるもの。
_RETRYABLE_BEFORE_FIRST_CHUNK = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


async def _open_stream(
    open_stream: Callable[[], Awaitable[AsyncIterator[Any]]],
) -> AsyncIterator[Any]:
    """ストリームを開き、最初のかたまりを受けてから、全部を流す反復子を返す。

    **最初のかたまりが届く前**に時間切れ・接続の失敗になったときだけ、
    `_FIRST_CHUNK_ATTEMPTS` 回まで開き直す（理由は `_REQUEST_TIMEOUT_SEC` の説明）。
    最初のかたまりを受けたあとの失敗は、そのまま呼び出し元に上げる。
    """
    for attempt in range(1, _FIRST_CHUNK_ATTEMPTS + 1):
        try:
            iterator = (await open_stream()).__aiter__()
            first = await iterator.__anext__()
            break
        except StopAsyncIteration:
            return _empty()
        except _RETRYABLE_BEFORE_FIRST_CHUNK as exc:
            if attempt >= _FIRST_CHUNK_ATTEMPTS:
                raise
            logger.warning(
                "[vertex] no response before the first chunk (%s); retrying (%d/%d)",
                type(exc).__name__,
                attempt + 1,
                _FIRST_CHUNK_ATTEMPTS,
            )
    return _prepend(first, iterator)


async def _prepend(first: Any, rest: AsyncIterator[Any]) -> AsyncIterator[Any]:
    yield first
    async for chunk in rest:
        yield chunk


async def _empty() -> AsyncIterator[Any]:
    return
    yield  # noqa: unreachable — async generator にするため


def thought_signature_kwargs(signatures: Mapping[str, str | None]) -> dict[str, Any]:
    """流して受けたツール呼び出しを、履歴に載せられる形にするための additional_kwargs。

    `{tool_call_id: base64 の署名}` を渡す。Gemini 3 は履歴の関数呼び出しに model が
    発行した thought_signature が無いと弾く。LangChain が読む場所・持ち方に合わせるため、
    その私的なキー名はこの層に閉じ込める（呼び出し側が langchain_google_genai の内部を
    知らなくて済むように）。
    """
    present = {call_id: sig for call_id, sig in signatures.items() if sig}
    if not present:
        return {}
    from langchain_google_genai.chat_models import (  # noqa: PLC0415
        _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
    )

    return {_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY: present}



# ── 引数の組み立て（vertex） ────────────────────────────────────────────────


_SEGMENT = re.compile(r"([^.\[\]]+)|\[(\d*)\]")


def _parse_path(json_path: str) -> list[str | int]:
    """`$.message` → ["message"]、`$.update_variables[0].name` → ["update_variables", 0, "name"]。

    **入れ子をそのまま持つ。** 引数はスカラーと文字列の配列だけではない。オブジェクトの
    配列（`complete_task` の `update_variables` は `{name, value}` の並び）が来るので、
    ここで平らにすると値が復元できず、呼び出し側には「`update_variables[0].name` という
    名前の引数」が届いてしまう（変数が一切設定されなくなる）。

    要素ではないもの（配列の終わりの合図 `$.a[]`、読めない path）は空リストを返す。
    """
    path = json_path[2:] if json_path.startswith("$.") else json_path
    if not path:
        return []
    out: list[str | int] = []
    pos = 0
    for m in _SEGMENT.finditer(path):
        gap = path[pos:m.start()]
        if gap not in ("", "."):
            return []
        pos = m.end()
        name, index = m.group(1), m.group(2)
        if name is not None:
            out.append(name)
        elif index == "":
            return []  # `[]` = 配列の終わりの合図。要素ではない
        else:
            out.append(int(index))
    return out if pos == len(path) else []


def _fragment_value(part: Any) -> Any:
    for attr in ("string_value", "bool_value", "number_value"):
        value = getattr(part, attr, None)
        if value is not None:
            return value
    return None


def _assign(root: dict, path: list[str | int], value: Any) -> None:
    """`path` の位置に値を置く。途中は dict で作る（配列は最後に直す）。"""
    node = root
    for segment in path[:-1]:
        child = node.get(segment)
        if not isinstance(child, dict):
            child = {}
            node[segment] = child
        node = child
    node[path[-1]] = value


def _to_lists(node: Any) -> Any:
    """添字で作った dict を list に直す。`{0: a, 1: b}` → `[a, b]`。"""
    if not isinstance(node, dict):
        return node
    converted = {k: _to_lists(v) for k, v in node.items()}
    if converted and all(isinstance(k, int) for k in converted):
        return [converted[k] for k in sorted(converted)]
    return converted


class ArgAssembler:
    """`partial_args` の断片を組み立てて、引数の dict に戻す。

    **1 つのツール呼び出しにつき 1 個**持つ。使い回すと別の呼び出しの引数が混ざる。

    断片は「jsonPath + 型付きの値」で来る。同じ path の文字列は順に連結し、それ以外は
    そのまま持つ。path の形（`a` / `a[0]` / `a[0].b`）をそのまま構造として復元するので、
    オブジェクトの配列も元の形に戻る。
    """

    def __init__(self) -> None:
        # path -> 値。文字列だけは連結するので別に持つ。
        self._values: dict[tuple, Any] = {}
        self._strings: dict[tuple, list[str]] = {}

    def feed(self, partial_args: list[Any]) -> list[tuple[str, int | None, str]]:
        """断片を食わせ、文字列として増えたぶんを (key, item_index, delta) で返す。

        返すのは**顧客に見せうる引数だけ**——トップレベルの文字列（`a`）と文字列の配列の
        要素（`a[0]`）。それより深いものは本文ではないので流さない。
        """
        deltas: list[tuple[str, int | None, str]] = []
        for part in partial_args or []:
            path = _parse_path(getattr(part, "json_path", "") or "")
            if not path:
                continue
            value = _fragment_value(part)
            if value is None:
                continue
            key = tuple(path)
            if isinstance(value, str):
                self._strings.setdefault(key, []).append(value)
                if not value:
                    continue
                if len(path) == 1:
                    deltas.append((str(path[0]), None, value))
                elif len(path) == 2 and isinstance(path[1], int):
                    deltas.append((str(path[0]), path[1], value))
            else:
                self._values[key] = value
        return deltas

    def result(self) -> dict[str, Any]:
        root: dict = {}
        for path, value in self._values.items():
            _assign(root, list(path), value)
        for path, chunks in self._strings.items():
            _assign(root, list(path), "".join(chunks))
        assembled = _to_lists(root)
        return assembled if isinstance(assembled, dict) else {}


# ── Vertex (Gemini) ─────────────────────────────────────────────────────────


def _bare_model(name: str | None) -> str | None:
    if not name:
        return None
    return name.split("/", 1)[1] if "/" in name else name


def _vertex_usage(chunk: Any) -> tuple[int, int, int] | None:
    """(入力, 出力, 出力のうち考えるのに使った分) のトークン数。

    出力は、本文（candidates）と考えた分（thoughts）を足した数。料金はどちらも出力として
    かかるので、合計で記録する。考えた分は別にも返し、ログに出す（`add_usage` の
    thinking_tokens）。
    """
    meta = getattr(chunk, "usage_metadata", None)
    if meta is None:
        return None
    prompt = getattr(meta, "prompt_token_count", None) or 0
    candidates = getattr(meta, "candidates_token_count", None) or 0
    thoughts = getattr(meta, "thoughts_token_count", None) or 0
    if not prompt and not candidates:
        return None
    return int(prompt), int(candidates) + int(thoughts), int(thoughts)


class _VertexCalls:
    """届いた `function_call` を、どのツール呼び出しのものか振り分ける。

    振り分けの手がかりは 2 つ。`FunctionCall.id` は populated なら一意なので、あれば
    それで引く。無いときは「名前が来たら新しい呼び出しの始まり」で数える（ツール名は
    その呼び出しの 1 チャンク目に必ず来て、引数より先に確定する）。
    """

    def __init__(self) -> None:
        self.calls: list[StreamedToolCall] = []
        self.assemblers: list[ArgAssembler] = []
        self._by_id: dict[str, int] = {}
        self._current: int | None = None

    def _open(self, call_id: str | None) -> int:
        index = len(self.calls)
        self.calls.append(StreamedToolCall(call_index=index))
        self.assemblers.append(ArgAssembler())
        if call_id:
            self._by_id[call_id] = index
        self._current = index
        return index

    def resolve(self, function_call: Any) -> int:
        call_id = getattr(function_call, "id", None) or None
        if call_id and call_id in self._by_id:
            self._current = self._by_id[call_id]
            return self._current
        if getattr(function_call, "name", None):
            return self._open(call_id)
        if call_id:
            # id はあるが未知で、名前も無い。取りこぼさないよう新しい呼び出しとして開く。
            return self._open(call_id)
        if self._current is None:
            return self._open(None)
        return self._current


def _drop_parameters_description(cfg: Any) -> None:
    """ツール宣言の `parameters.description` を落とす。

    **これが残っていると Gemini が出力を終えられなくなる。**LangChain は関数の
    docstring を `FunctionDeclaration.description` と `parameters.description`
    （引数全体の Schema の説明）の両方に入れる。後者があるだけで、本文を書く問で
    生成が止まらなくなり、時間いっぱいチャンクを出し続ける。

    同じ入力（128 問の #275）で投げ分けた実測。

        そのまま                        30.3 秒・569 チャンクで打ち切り
        parameters.description を消す    2.8 秒・25 チャンクで完走
        関数の description を消す        31.0 秒・559 チャンクで打ち切り
        プロパティの description を消す  30.1 秒・563 チャンクで打ち切り

    説明そのものは `FunctionDeclaration.description` に残るので、モデルに渡る
    情報は減らない。
    """
    for tool in getattr(cfg, "tools", None) or []:
        for fn in getattr(tool, "function_declarations", None) or []:
            params = getattr(fn, "parameters", None)
            if params is not None:
                params.description = None


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
    """`api_key` / `base_url` は OpenAI 用なのでここでは使わない。
    振り分け（`providers.resolve_tool_call_streamer`）が同じ形で呼べるよう受けている。

    `allowed_tool_names` を渡すと、そのターンに呼べる関数をその名前だけに絞る。
    **宣言（tools）は絞らない。** Gemini は宣言が変わるとプロンプトのキャッシュが効かなく
    なるので、絞るのは `allowed_function_names` の側でやる。

    `bounded=False` なら、上限（`_REQUEST_TIMEOUT_SEC`）も送り直し（`_open_stream`）も
    使わず、SDK の既定で待つ。1 ターンに時間がかかる前提の呼び出し用。

    `location` を渡すと、環境変数 GOOGLE_CLOUD_LOCATION より優先してその宛先で呼ぶ。

    `priority=True` なら優先受付（Priority PayGo）で呼ぶ。優先の枠を超えると Vertex は
    標準の受付に落として処理する（エラーにはならない）ので、落とされたら警告を出す。
    """
    from google.genai import types

    bare = _bare_model(model)
    from ..llm import get_llm  # 遅延 import（llm.py との循環を避ける）

    # `_prepare_request` が timeout と max_retries を http_options に入れてくれるので、
    # `generate_content_stream` に直接渡さなくても効く。bounded=False では渡さない
    # （SDK の既定 = 打ち切りなし）。
    limits = (
        {"timeout": _REQUEST_TIMEOUT_SEC, "max_retries": _REQUEST_ATTEMPTS} if bounded else {}
    )
    llm = get_llm(
        "vertex",
        bare,
        temperature=temperature,
        effort=effort,
        # 使用量はこのモジュールが記録する。LangChain のコールバックは astream を
        # 通らないので発火しない。
        track_usage=False,
        max_tokens=_MAX_OUTPUT_TOKENS,
        location=location,
        priority=priority,
        **limits,
    )

    # 引数の断片を受ける設定。tool_choice ではなく tool_config に直接渡す
    # （tool_choice 経由だと FunctionCallingConfig が作り直され、フラグが落ちる）。
    function_calling_config: dict[str, Any] = {
        "mode": "ANY",
        "stream_function_call_arguments": True,
    }
    if allowed_tool_names:
        function_calling_config["allowed_function_names"] = list(allowed_tool_names)
    request = llm._prepare_request(  # noqa: SLF001 - 組み立てだけ借りる（説明は冒頭）
        messages,
        tools=tools,
        tool_config={"function_calling_config": function_calling_config},
    )
    # 引数を分割して受ける設定は、SDK が勝手に関数を実行する仕組み（AFC）と併用できない。
    # ここではツールを実行しないので切って構わない。
    cfg = request.get("config")
    if cfg is not None:
        cfg.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        _drop_parameters_description(cfg)

    tracked = _VertexCalls()
    result = StreamedResult()
    usage: tuple[int, int, int] | None = None
    # どの受付で処理されたか（標準 / 優先）。応答の usage_metadata に載る。
    traffic_type: Any = None

    # 応答が遅いとき、「モデルが長く考えていた」のか「Vertex で待たされた」のかを分けるため、
    # 最初のかたまりまでの秒数と全体の秒数を測り、考えた分のトークン数と一緒にログに出す。
    # bounded のときは、最初のかたまりの前に時間切れになると `_open_stream` が開き直すので、
    # first_chunk_seconds には打ち切られた試行の待ちも含まれる（開き直したことは、その警告で
    # ログに出る）。bounded=False では開き直さないので、1 回分の待ちそのもの。
    started = time.monotonic()

    started = time.monotonic()
    first_chunk_seconds: float | None = None
    open_stream = lambda: llm.client.aio.models.generate_content_stream(**request)  # noqa: E731
    chunks = await (_open_stream(open_stream) if bounded else open_stream())
    async for chunk in chunks:
        if first_chunk_seconds is None:
            first_chunk_seconds = time.monotonic() - started
        seen = _vertex_usage(chunk)
        if seen is not None:
            usage = seen
        traffic_type = getattr(getattr(chunk, "usage_metadata", None), "traffic_type", None) or traffic_type
        for candidate in chunk.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    text = getattr(part, "text", None)
                    if text and not getattr(part, "thought", False):
                        result.text += text
                    continue

                index = tracked.resolve(function_call)
                call = tracked.calls[index]

                signature = getattr(part, "thought_signature", None)
                if signature and call.thought_signature is None:
                    call.thought_signature = (
                        base64.b64encode(signature).decode()
                        if isinstance(signature, bytes)
                        else str(signature)
                    )

                name = getattr(function_call, "name", None)
                if name and call.name is None:
                    call.name = name
                    yield "name", (index, name)

                if function_call.partial_args:
                    for key, item_index, text in tracked.assemblers[index].feed(
                        function_call.partial_args
                    ):
                        yield "delta", ArgDelta(index, key, item_index, text)
                elif function_call.args:
                    # フラグが効かなかった場合（完成形で 1 個だけ届いた）。
                    # 流れないだけで回答自体は作れるので、まとめて 1 つの delta にする。
                    call.args = dict(function_call.args)
                    body_arg = body_arg_for(call.name or "") if body_arg_for else None
                    body = call.args.get(body_arg) if body_arg else None
                    if isinstance(body, str) and body:
                        yield "delta", ArgDelta(index, body_arg, None, body)

    for call, assembler in zip(tracked.calls, tracked.assemblers):
        assembled = assembler.result()
        if assembled:
            call.args = {**call.args, **assembled}
    result.calls = tracked.calls
    if priority and traffic_type not in (None, types.TrafficType.ON_DEMAND_PRIORITY):
        logger.warning("[vertex] priority was requested but served as %s (model=%s)", traffic_type, bare)
    if track_usage and usage:
        add_usage(
            "vertex",
            bare or "",
            usage[0],
            usage[1],
            elapsed_time=time.monotonic() - started,
            thinking_tokens=usage[2],
            first_chunk_seconds=first_chunk_seconds,
        )
    yield "done", result


