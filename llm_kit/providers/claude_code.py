"""
Claude Code CLI を LangChain ChatModel として叩くプロバイダー

`claude -p --output-format json` をサブプロセスで実行し、レスポンスを
LangChain の AIMessage に変換する。これにより、他プロバイダー（openai/anthropic/
google/vertex）と全く同じインターフェース（invoke / ainvoke / stream /
with_structured_output 等）で Claude Code を「API のように」呼び出せる。

特徴:
    - OAuth ログイン認証をそのまま利用（ANTHROPIC_API_KEY 不要）
    - Claude Code のエージェント機能（ツール実行・ファイル読み書き等）も利用可能
    - 構造化出力は CLI の --json-schema を使用
    - 実コスト（total_cost_usd）を usage_tracker に記録

Example:
    from llm_kit import get_llm, invoke

    # シンプル
    text = invoke("こんにちは", provider="claude_code")

    # ツールを使わせない（純粋な LLM 応答）
    llm = get_llm("claude_code", disallowed_tools=["Bash", "Edit", "Write"])

    # エージェントとして使う（ファイル探索などを許可）
    llm = get_llm(
        "claude_code",
        permission_mode="bypassPermissions",
        cwd="/path/to/project",
    )
"""

import asyncio
import json
import logging
import shutil
import subprocess
import time
from typing import Any, Optional, Sequence, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field

from .. import usage_tracker
from ..exceptions import LLMAPIError, LLMConfigError, LLMResponseError

logger = logging.getLogger(__name__)

PROVIDER_NAME = "claude_code"


def _extract_text(content: Any) -> str:
    """メッセージ content（str または content blocks）からテキストを抽出"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _clean_json_schema(schema: dict) -> dict:
    """Pydantic 由来の JSON Schema から CLI に不要なキーを除去"""
    schema = dict(schema)
    schema.pop("title", None)
    return schema


class ChatClaudeCode(BaseChatModel):
    """Claude Code CLI をバックエンドにした LangChain ChatModel"""

    # pydantic が `model_` 名前空間を保護するため無効化（`model` フィールドを使う）
    model_config = ConfigDict(protected_namespaces=())

    model: str = "claude-sonnet-4-6"
    """使用モデル。エイリアス（sonnet/opus）または正式名（claude-sonnet-4-6 等）"""

    cli_path: str = "claude"
    """claude CLI の実行パス"""

    # --- システムプロンプト ---
    system_prompt: Optional[str] = None
    """Claude Code の既定システムプロンプトに追記する（--append-system-prompt）"""

    override_system_prompt: Optional[str] = None
    """Claude Code の既定システムプロンプトを完全に置き換える（--system-prompt）"""

    # --- ツール / 権限 ---
    allowed_tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    tools: Optional[str] = None
    """利用可能ツールを限定（"" で全無効、"default" で全有効、"Bash,Read" 等）"""
    permission_mode: Optional[str] = None
    """default / acceptEdits / auto / bypassPermissions / dontAsk / plan"""
    dangerously_skip_permissions: bool = False
    add_dirs: Optional[list[str]] = None

    # --- 実行制御 ---
    cwd: Optional[str] = None
    """CLI を実行する作業ディレクトリ"""
    timeout: float = 600.0
    max_budget_usd: Optional[float] = None
    session_persistence: bool = False
    """False の場合 --no-session-persistence を付与（API 的な使い捨て実行）"""
    extra_args: list[str] = Field(default_factory=list)
    """追加で渡す任意の CLI 引数"""

    # --- 内部用 ---
    track_usage: bool = True
    json_schema: Optional[dict] = None
    """構造化出力スキーマ（with_structured_output 経由で設定）"""

    @property
    def _llm_type(self) -> str:
        return PROVIDER_NAME

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "provider": PROVIDER_NAME}

    # ------------------------------------------------------------------
    # コマンド構築
    # ------------------------------------------------------------------
    def _messages_to_prompt(self, messages: list[BaseMessage]) -> tuple[list[str], str]:
        """メッセージ列を (システムプロンプト群, プロンプト文字列) に変換"""
        system_parts: list[str] = []
        convo: list[tuple[str, str]] = []

        for m in messages:
            text = _extract_text(m.content)
            if isinstance(m, SystemMessage):
                if text:
                    system_parts.append(text)
            elif isinstance(m, HumanMessage):
                convo.append(("Human", text))
            elif isinstance(m, AIMessage):
                convo.append(("Assistant", text))
            else:
                convo.append((getattr(m, "type", "Human").capitalize(), text))

        if len(convo) == 1 and convo[0][0] == "Human":
            prompt = convo[0][1]
        else:
            prompt = "\n\n".join(f"{role}: {content}" for role, content in convo)

        return system_parts, prompt

    def _build_args(self, system_parts: list[str]) -> list[str]:
        """プロンプト以外の CLI 引数リストを構築"""
        args = [
            self.cli_path,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
        ]

        if self.override_system_prompt is not None:
            args += ["--system-prompt", self.override_system_prompt]

        append_parts = list(system_parts)
        if self.system_prompt:
            append_parts.insert(0, self.system_prompt)
        if append_parts:
            args += ["--append-system-prompt", "\n\n".join(append_parts)]

        if self.json_schema is not None:
            args += ["--json-schema", json.dumps(self.json_schema, ensure_ascii=False)]

        if self.allowed_tools:
            args += ["--allowedTools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            args += ["--disallowedTools", ",".join(self.disallowed_tools)]
        if self.tools is not None:
            args += ["--tools", self.tools]
        if self.permission_mode:
            args += ["--permission-mode", self.permission_mode]
        if self.dangerously_skip_permissions:
            args += ["--dangerously-skip-permissions"]
        if self.add_dirs:
            args += ["--add-dir", *self.add_dirs]
        if self.max_budget_usd is not None:
            args += ["--max-budget-usd", str(self.max_budget_usd)]
        if not self.session_persistence:
            args += ["--no-session-persistence"]

        args += list(self.extra_args)
        return args

    # ------------------------------------------------------------------
    # レスポンス解析
    # ------------------------------------------------------------------
    def _parse_payload(self, stdout: str) -> dict:
        stdout = stdout.strip()
        if not stdout:
            raise LLMResponseError("Claude Code CLI returned empty output")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                # 出力の中身は利用者の発言を引くことがあるので、長さだけを出す。
                f"Failed to parse Claude Code CLI JSON output: {e} | raw_len={len(stdout)}"
            )

        if payload.get("is_error") or payload.get("subtype") != "success":
            # result はモデルの出力（利用者の発言を引くことがある）なので、種類と長さだけを出す。
            raise LLMAPIError(
                f"Claude Code CLI error: subtype={payload.get('subtype')} "
                f"result_len={len(str(payload.get('result') or ''))}",
                provider=PROVIDER_NAME,
            )
        return payload

    def _build_result(self, payload: dict, elapsed: float) -> ChatResult:
        text = payload.get("result", "") or ""
        usage = payload.get("usage", {}) or {}

        input_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        output_tokens = usage.get("output_tokens", 0)
        cost_usd = payload.get("total_cost_usd", 0.0) or 0.0

        if self.track_usage and (input_tokens or output_tokens or cost_usd):
            usage_tracker.add_usage(
                provider=PROVIDER_NAME,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                elapsed_time=elapsed,
                cost_usd=cost_usd,
            )

        additional_kwargs: dict[str, Any] = {}
        if "structured_output" in payload:
            additional_kwargs["structured_output"] = payload["structured_output"]

        message = AIMessage(
            content=text,
            additional_kwargs=additional_kwargs,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata={
                "model": self.model,
                "total_cost_usd": cost_usd,
                "session_id": payload.get("session_id"),
                "num_turns": payload.get("num_turns"),
                "stop_reason": payload.get("stop_reason"),
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(
            generations=[generation],
            llm_output={"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
        )

    # ------------------------------------------------------------------
    # 同期実行
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if shutil.which(self.cli_path) is None and "/" not in self.cli_path:
            raise LLMConfigError(
                f"Claude Code CLI not found: '{self.cli_path}'. Install it or set cli_path."
            )

        system_parts, prompt = self._messages_to_prompt(messages)
        args = self._build_args(system_parts)

        start = time.time()
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise LLMAPIError(
                f"Claude Code CLI timed out after {self.timeout}s", provider=PROVIDER_NAME
            )
        elapsed = time.time() - start

        if proc.returncode != 0:
            raise LLMAPIError(
                f"Claude Code CLI exited with code {proc.returncode}: {proc.stderr.strip()[:500]}",
                provider=PROVIDER_NAME,
                status_code=proc.returncode,
            )

        payload = self._parse_payload(proc.stdout)
        return self._build_result(payload, elapsed)

    # ------------------------------------------------------------------
    # 非同期実行
    # ------------------------------------------------------------------
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if shutil.which(self.cli_path) is None and "/" not in self.cli_path:
            raise LLMConfigError(
                f"Claude Code CLI not found: '{self.cli_path}'. Install it or set cli_path."
            )

        system_parts, prompt = self._messages_to_prompt(messages)
        args = self._build_args(system_parts)

        start = time.time()
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise LLMAPIError(
                f"Claude Code CLI timed out after {self.timeout}s", provider=PROVIDER_NAME
            )
        elapsed = time.time() - start

        if proc.returncode != 0:
            raise LLMAPIError(
                f"Claude Code CLI exited with code {proc.returncode}: "
                f"{stderr_b.decode('utf-8', 'replace').strip()[:500]}",
                provider=PROVIDER_NAME,
                status_code=proc.returncode,
            )

        payload = self._parse_payload(stdout_b.decode("utf-8", "replace"))
        return self._build_result(payload, elapsed)

    # ------------------------------------------------------------------
    # 構造化出力
    # ------------------------------------------------------------------
    def with_structured_output(
        self,
        schema: Union[dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """CLI の --json-schema を使った構造化出力

        schema が Pydantic BaseModel ならインスタンスを、dict（JSON Schema）なら
        dict を返す Runnable を生成する。
        """
        is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)
        if is_pydantic:
            json_schema = _clean_json_schema(schema.model_json_schema())
        elif isinstance(schema, dict):
            json_schema = _clean_json_schema(schema)
        else:
            raise LLMConfigError(
                "with_structured_output には Pydantic BaseModel または JSON Schema dict を渡してください"
            )

        bound = self.model_copy(update={"json_schema": json_schema})

        def _parse(message: AIMessage):
            data = message.additional_kwargs.get("structured_output")
            if data is None:
                # フォールバック: content を JSON として解釈
                from ..utils import parse_json

                data = parse_json(_extract_text(message.content))
            return schema.model_validate(data) if is_pydantic else data

        parser = RunnableLambda(_parse)

        if include_raw:
            def _with_raw(message: AIMessage):
                try:
                    return {"raw": message, "parsed": _parse(message), "parsing_error": None}
                except Exception as e:  # noqa: BLE001
                    return {"raw": message, "parsed": None, "parsing_error": e}

            return bound | RunnableLambda(_with_raw)

        return bound | parser
