"""LangChain ChatModel の取得と呼び出し（メインエントリーポイント）。

`get_llm` がモデルを作り、`invoke` / `ainvoke` / `stream` / `astream` が
プロンプト 1 本を渡して文字列を受ける。`astream_tool_calls` はツール呼び出しを
強制して、その**引数**を分割して受けながら流す（顧客に見せる本文をツール引数に
入れている呼び出し側のため）。返る型は `types.py`。
"""

import asyncio
import logging
import warnings
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# LangChain の非推奨警告を抑制
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_google_vertexai")
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .config import DEFAULT_PROVIDER
from .providers import resolve_tool_call_streamer
from .types import BodyArgResolver
from .utils import get_model
from .usage_tracker import create_usage_callback


# === グローバルセマフォ（LLM同時呼び出し数制御） ===
_llm_semaphore: asyncio.Semaphore | None = None


def configure_llm_concurrency(max_concurrent: int = 10) -> None:
    """
    LLM呼び出しの同時実行数を設定する

    Args:
        max_concurrent: 最大同時実行数（デフォルト10）

    Note:
        この関数は非同期処理を開始する前に一度だけ呼び出すこと。
        設定後、ainvoke_structured等の非同期LLM呼び出しは
        このセマフォで同時実行数が制限される。
    """
    global _llm_semaphore
    _llm_semaphore = asyncio.Semaphore(max_concurrent)


def get_llm_semaphore() -> asyncio.Semaphore | None:
    """現在のLLMセマフォを取得"""
    return _llm_semaphore


# 「どれくらい考えさせるか」の統一のつまみ。provider ごとに名前も形も違うので、
# ここで 1 つの語彙にまとめて get_llm の入口で振り分ける。
#
#   vertex / google … reasoning_effort（LangChain は thinking_level も同じ意味の
#                     別名として受ける）。
#   openai          … reasoning_effort
#   anthropic       … output_config.effort（xhigh / max を持つ）
#
# 語彙は各 provider の和集合。弱い順に並べてある（この順序で丸める）。
EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# provider ごとに実際に受け取れる値。和集合を持っているのはこの層だけなので、
# 「持っていない値が来たらどうするか」もここで決める。呼び出し側に provider ごとの
# 対応表を書かせると、役割の設定が provider の都合で汚れる。
SUPPORTED_EFFORTS = {
    "vertex": ("none", "minimal", "low", "medium", "high"),
    "google": ("none", "minimal", "low", "medium", "high"),
    "openai": ("none", "low", "medium", "high", "xhigh"),
    "anthropic": ("low", "medium", "high", "xhigh", "max"),
}


def _normalize_effort(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if v not in EFFORT_LEVELS:
        logger.warning("未知の effort を無視します: %r (有効: %s)", value, ", ".join(EFFORT_LEVELS))
        return None
    return v


def _clamp_effort(effort: str | None, provider: str) -> str | None:
    """その provider が持たない effort を、持っている中の**直下**の値へ落とす。

    下へ丸めるのは、頼まれた以上に考えさせないため。上へ丸めると、速さを見たくて
    minimal を指定した測定が黙って遅くなる。provider の最小値より下しか無い場合だけ、
    その最小値へ上げる（例: anthropic に none を渡す）。

    例: minimal -> openai は none、max -> vertex は high。
    """
    if effort is None:
        return None
    supported = SUPPORTED_EFFORTS.get(provider)
    if not supported or effort in supported:
        return effort
    order = EFFORT_LEVELS.index(effort)
    below = [s for s in supported if EFFORT_LEVELS.index(s) <= order]
    picked = max(below, key=EFFORT_LEVELS.index) if below else min(supported, key=EFFORT_LEVELS.index)
    logger.info(
        "%s は effort=%r を持たないので %r に丸めます (対応: %s)",
        provider, effort, picked, ", ".join(supported),
    )
    return picked


def get_llm(
    provider: str = None,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    track_usage: bool = True,
    *,
    effort: str | None = None,
    thinking_level: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """
    LangChain ChatModelを取得

    Args:
        provider: openai / anthropic / google / vertex / bedrock / claude_code
            （省略時は config.DEFAULT_PROVIDER）
        model: モデル名（省略時は config.DEFAULT_MODELS）
        temperature: 温度パラメータ
        max_tokens: 最大トークン数（None ならモデル既定に任せる）
        track_usage: 使用量トラッキング有効化
        effort: どれくらい考えさせるか（EFFORT_LEVELS）。provider が持たない値は直下へ丸める
        thinking_level: effort の古い名前
        **kwargs: provider の ChatModel にそのまま渡す

    Example:
        llm = get_llm("vertex", "gemini-3-flash-preview")
        print(llm.invoke("Hello").text)

        # LangGraph にもそのまま渡せる
        def call_model(state):
            return {"messages": [get_llm("anthropic", "claude-sonnet-5").invoke(state["messages"])]}
    """
    provider = provider or DEFAULT_PROVIDER
    model = get_model(provider, model)

    # 「どれくらい考えさせるか」は provider ごとに名前が違う。ここで 1 つに寄せ、
    # 各分岐は自分の形に落とすだけにする。thinking_level は従来の名前で、effort の別名。
    effort = _clamp_effort(_normalize_effort(effort or thinking_level), provider)

    # claude_code は CLI が実コストを返すため、自前で usage_tracker に記録する
    # （PRICING ベースのコールバックは使わない）
    if provider == "claude_code":
        from .providers.claude_code import ChatClaudeCode

        if effort is not None:
            logger.info("claude_code の思考量は CLI 側の設定なので effort は無視します (effort=%r)", effort)
        kwargs.pop("callbacks", None)
        return ChatClaudeCode(
            model=model,
            track_usage=track_usage,
            **kwargs,
        )

    # コールバック設定
    callbacks = kwargs.pop("callbacks", [])
    if track_usage:
        callbacks.append(create_usage_callback(provider, model))

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if effort is not None:
            kwargs.setdefault("reasoning_effort", effort)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            **kwargs,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Anthropic は output_config.effort（low/medium/high/xhigh/max）と
        # thinking={"type": "adaptive"} の組み合わせ。langchain_anthropic の版によって
        # 受け口が違うため、ここでは渡さず落としたことを知らせる。
        if effort is not None:
            logger.info("anthropic には effort をまだ渡していません (effort=%r, model=%s)", effort, model)
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            **kwargs,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from .utils import get_google_credentials

        # サービスアカウント認証（GOOGLE_APPLICATION_CREDENTIALS）を使用
        credentials, project_id = get_google_credentials()

        if effort is not None:
            kwargs.setdefault("reasoning_effort", effort)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            credentials=credentials,
            project=project_id,
            callbacks=callbacks,
            **kwargs,
        )

    elif provider == "bedrock":
        import os
        from langchain_aws import ChatBedrockConverse

        # 認証は環境変数のベアラトークン（AWS_BEARER_TOKEN_BEDROCK）または
        # 標準AWS認証（プロファイル/アクセスキー/SSO）を boto3 が自動解決する。
        region = kwargs.pop("region_name", None) or os.environ.get("AWS_REGION", "ap-northeast-1")
        if effort is not None:
            logger.info("bedrock には effort をまだ渡していません (effort=%r, model=%s)", effort, model)

        return ChatBedrockConverse(
            model=model,
            region_name=region,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            **kwargs,
        )

    elif provider == "vertex":
        # `langchain-google-vertexai` / ChatVertexAI から `langchain-google-genai`
        # （統合 google-genai SDK）＋ vertexai=True に更新。Gemini 3 系にネイティブの
        # `reasoning_effort` を渡せるのはこちら。認証は ADC
        # （GOOGLE_APPLICATION_CREDENTIALS）。
        from langchain_google_genai import ChatGoogleGenerativeAI
        from .utils import get_vertex_config

        vertex = get_vertex_config()

        params: dict = {
            "model": model,
            "temperature": temperature,
            "vertexai": True,
            "project": vertex["project_id"],
            # location を渡されたら、環境変数より優先する（役割ごとに宛先を変えるため）。
            "location": kwargs.pop("location", None) or vertex["location"],
            "callbacks": callbacks,
        }
        # 上限を渡さない呼び出しでモデル既定を潰さないよう、指定時だけ設定する
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        # LangChain は thinking_level を reasoning_effort の別名として受けるので、
        # 統一のつまみ（effort）をそのまま渡す。
        if effort is not None:
            params["reasoning_effort"] = effort

        # 優先受付（Priority PayGo）はヘッダーで頼む。google-genai の
        # GenerateContentConfig.service_tier は Vertex に 400 で弾かれる（2.18.1 で実測）。
        # クライアントに付けるので、このモデルからの呼び出しは全部（流す呼び出しも）優先になる。
        if kwargs.pop("priority", False):
            kwargs["additional_headers"] = {
                **(kwargs.get("additional_headers") or {}),
                "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
            }

        return ChatGoogleGenerativeAI(**params, **kwargs)

    else:
        from .exceptions import LLMConfigError
        raise LLMConfigError(f"Unknown provider: {provider}")


# ═══════════════════════════════════════════════════════════════════════════
# プロンプト 1 本で呼ぶ
#
# どれも [SystemMessage(system_prompt)?, HumanMessage(prompt)] を送り、返事は
# `.text`（文字のブロックだけを繋いだ文字列）で受ける。Gemini 3 などは content を
# ブロックの list で返すので、`.content` のままだと文字列にならない。
# 非同期の呼び出しは configure_llm_concurrency の上限の中で待つ。
# ═══════════════════════════════════════════════════════════════════════════


def _messages(prompt: str, system_prompt: str | None) -> list[BaseMessage]:
    head = [SystemMessage(content=system_prompt)] if system_prompt else []
    return [*head, HumanMessage(content=prompt)]


async def _limited(awaitable):
    """configure_llm_concurrency で決めた上限の中で awaitable を待つ。"""
    if _llm_semaphore is None:
        return await awaitable
    async with _llm_semaphore:
        return await awaitable


def _retry_temperature(temperature: float, attempt: int) -> float:
    """送り直すたびに温度を 0.1 ずつ上げる（上限 1.0）。同じ崩れ方を繰り返させないため。"""
    return temperature if attempt == 0 else min(temperature + 0.1 * attempt, 1.0)


def get_structured_llm(
    schema,
    provider: str = None,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
    **kwargs,
):
    """
    Structured Output用のLLMを取得

    Args:
        schema: Pydantic BaseModel または dict (JSON Schema)
        provider, model, temperature, max_tokens, track_usage, **kwargs: get_llm と同じ

    Returns:
        Runnable: schema型を返すRunnable

    Example:
        class Colors(BaseModel):
            colors: list[str]

        llm = get_structured_llm(Colors, "openai", "gpt-5.2")
        print(llm.invoke("3つの色を返して").colors)
    """
    llm = get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        track_usage=track_usage,
        **kwargs,
    )
    return llm.with_structured_output(schema)


def invoke(
    prompt: str,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
) -> str:
    """
    プロンプト 1 本を送り、返事の文字列を返す

    Example:
        text = invoke("Hello!", provider="vertex", model="gemini-3-flash-preview")
    """
    llm = get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        track_usage=track_usage,
    )
    return llm.invoke(_messages(prompt, system_prompt)).text


def invoke_structured(
    prompt: str,
    schema,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
):
    """
    プロンプト 1 本を送り、schema 型のインスタンスを返す（LangChain の with_structured_output）

    Example:
        class Answer(BaseModel):
            answer: str
            confidence: float

        result = invoke_structured("What is 2+2?", Answer, provider="openai", model="gpt-5.2")
    """
    llm = get_structured_llm(
        schema=schema,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        track_usage=track_usage,
    )
    return llm.invoke(_messages(prompt, system_prompt))


async def ainvoke(
    prompt: str,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
) -> str:
    """invoke の非同期版"""
    llm = get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        track_usage=track_usage,
    )
    response = await _limited(llm.ainvoke(_messages(prompt, system_prompt)))
    return response.text


async def ainvoke_structured(
    prompt: str,
    schema,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
    max_retries: int = 3,
):
    """
    Structured Output呼び出し（非同期版） - LangChain標準パース

    LangChain の with_structured_output（json_mode）に読ませる。読めなければ温度を
    上げて、同じ文で送り直す。誤りを添えて送り直すなら ainvoke_structured_with_retry。

    Raises:
        LLMResponseError: max_retries 回送っても読めなかった場合
    """
    from .exceptions import LLMResponseError

    messages = _messages(prompt, system_prompt)

    async def _invoke_with_retry():
        last_error_msg = None

        for attempt in range(max_retries):
            llm = get_llm(
                provider=provider,
                model=model,
                temperature=_retry_temperature(temperature, attempt),
                max_tokens=max_tokens,
                track_usage=track_usage,
            )

            structured_llm = llm.with_structured_output(schema, include_raw=True, method="json_mode")
            result = await structured_llm.ainvoke(messages)

            parsed = result.get("parsed") if result else None
            if parsed is not None:
                return parsed

            raw_message = result.get("raw")
            raw_content = raw_message.content if raw_message else None
            parsing_error = result.get("parsing_error")

            # モデルの出力と、それを含みうる parsing_error の文は、利用者の発言を引くことが
            # あるのでメッセージに入れない。種類と長さだけにする。
            last_error_msg = "Structured output parsing failed"
            if parsing_error:
                last_error_msg += f" | Error: {type(parsing_error).__name__}"
            last_error_msg += f" | raw_len={len(str(raw_content)) if raw_content else 0}"

            if attempt < max_retries - 1:
                await asyncio.sleep(2.0)

        raise LLMResponseError(last_error_msg)

    return await _limited(_invoke_with_retry())


async def ainvoke_structured_with_retry(
    prompt: str,
    schema,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
    max_retries: int = 3,
):
    """
    Structured Output呼び出し（非同期版） - ソフトバリデーション＋フィードバック付きリトライ

    LangChainのパースに頼らず、自前でバリデーション・修復・リトライを行う。

    リトライ戦略:
    1. スキーマを LLM の機能としては渡さず、文の末尾の指示（json_output_instruction）で JSON を書かせる
    2. 返事を緩く読み直す（try_lenient_parse: 前後の文・切れた JSON・名前・型を直す）
    3. それでも読めなければ、前回の出力と誤りを添えて、温度を上げて送り直す

    Raises:
        LLMResponseError: max_retries 回送っても読めなかった場合
    """
    from .exceptions import LLMResponseError
    from .utils import try_lenient_parse, build_retry_messages, json_output_instruction

    original_messages = _messages(prompt + json_output_instruction(schema), system_prompt)
    schema_name = getattr(schema, "__name__", str(schema))

    async def _invoke_with_retry():
        current_messages = original_messages
        error_detail = "出力がスキーマに適合しませんでした"
        raw = ""

        for attempt in range(max_retries):
            llm = get_llm(
                provider=provider,
                model=model,
                temperature=_retry_temperature(temperature, attempt),
                max_tokens=max_tokens,
                track_usage=track_usage,
            )

            response = await llm.ainvoke(current_messages)
            raw = response.text if response else ""

            parsed = try_lenient_parse(raw, schema)
            if parsed is not None:
                if attempt > 0:
                    logger.info("%s: attempt %d/%d で読めた", schema_name, attempt + 1, max_retries)
                return parsed

            # モデルの出力は利用者の発言を引くことがあるので、ログと例外には長さだけを出す。
            logger.warning(
                "%s: attempt %d/%d parse failed | raw_len=%d",
                schema_name, attempt + 1, max_retries, len(raw),
            )

            if attempt < max_retries - 1:
                current_messages = build_retry_messages(
                    original_messages=original_messages,
                    raw_output=raw,
                    error_detail=error_detail,
                    schema=schema,
                )
                await asyncio.sleep(2.0)

        raise LLMResponseError(
            f"Structured output parsing failed | Error: {error_detail} | raw_len={len(raw)}"
        )

    return await _limited(_invoke_with_retry())


def stream(
    prompt: str,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
):
    """
    プロンプト 1 本を送り、返事を届いた分ずつ文字列で返す

    Example:
        for text in stream("Tell me a story"):
            print(text, end="", flush=True)
    """
    llm = get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        track_usage=track_usage,
    )
    for chunk in llm.stream(_messages(prompt, system_prompt)):
        yield chunk.text


async def astream(
    prompt: str,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_usage: bool = True,
):
    """
    stream の非同期版

    Example:
        async for text in astream("Tell me a story"):
            print(text, end="", flush=True)
    """
    llm = get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        track_usage=track_usage,
    )
    async for chunk in llm.astream(_messages(prompt, system_prompt)):
        yield chunk.text


# ═══════════════════════════════════════════════════════════════════════════
# ツール呼び出しの引数を分割して受けながら流す
#
# 顧客に見せる本文をツール呼び出しの**引数**に入れている呼び出し側のために置いている。
# 引数が完成形で 1 個だけ届くと、生成が終わるまで利用者に何も出ない。断片で受ければ
# 最初の文字が出るまでが短くなる。
#
# provider ごとに届き方が違う（Gemini は jsonPath + 値、OpenAI は生の JSON の断片）。
# その差は providers/ が吸収し、ここは振り分けるだけ。yield する形は共通:
#
#     ("name",  (call_index, ツール名))   … その呼び出しの 1 チャンク目。引数より前に来る
#     ("delta", ArgDelta)                 … 文字列の引数が増えたぶん
#     ("done",  StreamedResult)           … 最後に 1 回
#
# **1 ターンに複数のツール呼び出し（並列ツール呼び出し）を扱う。** tool_choice="required"
# も mode="ANY" も「1 回以上」の指定で、回数を 1 に固定しない。
#
# どの引数が顧客向けの本文かはドメインの取り決めなので、ここでは持たない。
# 呼び出し側が body_arg_for で渡す。
# ═══════════════════════════════════════════════════════════════════════════


async def astream_tool_calls(
    messages: list[BaseMessage],
    tools: list[Any],
    *,
    provider: str | None = None,
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
    """ツール呼び出しを強制し、引数の断片を流しながら受ける。

    Args:
        messages: モデルに渡す履歴。
        tools: LangChain のツール。
        provider: "openai" なら OpenAI、それ以外は Vertex の Gemini。
        model: モデル名。`provider/model` の形でも受ける。
        temperature: 温度（vertex のみ。openai は送らない）。
        effort: どれくらい考えさせるか。provider ごとの名前の違いは get_llm が吸収する。
        api_key: OpenAI の API キー。省略時は OPENAI_API_KEY。
        base_url: OpenAI の宛先。省略時は本番。互換 API や試験用の宛先に向けるときに使う。
        body_arg_for: ツール名から顧客向けの本文が入る引数名を返す関数。
            OpenAI は本文の key を狙って抽出するので必須。Gemini は断片が jsonPath 付きで
            来るため、フラグが効かなかったときの保険にだけ使う。
        track_usage: 使用量を usage_tracker に記録するか。
        allowed_tool_names: このターンに呼べるツールをこの名前だけに絞る。省略時は
            渡したツールすべて。**絞り方は provider ごとに違う**（vertex は宣言を
            変えずに allowed_function_names、OpenAI は宣言そのものを絞る）。
        bounded: 待ち時間に上限を付けるか。vertex は 45 秒で打ち切り、最初のかたまりの前だけ
            1 回送り直す。OpenAI は 45 秒で打ち切り、SDK に送り直させない。False なら打ち切りも
            送り直しもせず、SDK の既定で待つ。1 ターンに時間がかかる前提の呼び出しは、
            打ち切るとかえって失敗するので False で呼ぶ。
        location: Vertex のロケーション（vertex のみ）。省略時は GOOGLE_CLOUD_LOCATION。
        priority: Vertex の優先受付（Priority PayGo）で呼ぶか（vertex のみ）。
    """
    stream = resolve_tool_call_streamer(provider)(
        messages,
        tools,
        model=model,
        temperature=temperature,
        effort=effort,
        api_key=api_key,
        base_url=base_url,
        body_arg_for=body_arg_for,
        track_usage=track_usage,
        allowed_tool_names=allowed_tool_names,
        bounded=bounded,
        location=location,
        priority=priority,
    )
    async for item in stream:
        yield item


