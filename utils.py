"""
ユーティリティ関数（Google の認証と宛先、モデルと料金の引き当て、Structured Output の緩い読み直し）
"""

import json
import logging
import os
import re
import types
import typing
from typing import Any, Optional, get_args, get_origin

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from .exceptions import LLMConfigError

logger = logging.getLogger(__name__)

# .envファイルを自動読み込み
load_dotenv()


# =============================================================================
# Google の認証と宛先
# =============================================================================
#
# project の決め方は google / vertex で同じ。VERTEX_PROJECT_ID > GOOGLE_CLOUD_PROJECT >
# サービスアカウントの JSON に書いてある project_id > ADC（gcloud の設定）の順。

_GOOGLE_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _service_account_path() -> str | None:
    """GOOGLE_APPLICATION_CREDENTIALS が指すファイル。指していなければ None（ADC を使う）。"""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path and not os.path.exists(path):
        raise LLMConfigError(f"Service account JSON file not found: {path}")
    return path


def _project_in_file(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            return json.load(f).get("project_id")
    except (json.JSONDecodeError, OSError):
        return None


def _project_in_env() -> str | None:
    return os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _project_not_found() -> LLMConfigError:
    return LLMConfigError(
        "Project ID not found. Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT, "
        "or ensure gcloud is configured with a default project."
    )


_google_credentials_cache: tuple | None = None


def get_google_credentials():
    """
    Google GenAI 用の (credentials, project_id) を返す（キャッシュ付き）。

    並列に呼ばれても、認証の問い合わせは初回の 1 回だけにする。
    認証は GOOGLE_APPLICATION_CREDENTIALS のサービスアカウント JSON、無ければ ADC
    （gcloud auth application-default login）。
    """
    global _google_credentials_cache

    if _google_credentials_cache is not None:
        return _google_credentials_cache

    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    from google.oauth2 import service_account

    path = _service_account_path()
    if path:
        credentials = service_account.Credentials.from_service_account_file(path, scopes=_GOOGLE_SCOPES)
        project_in_auth = _project_in_file(path)
    else:
        try:
            credentials, project_in_auth = google.auth.default(scopes=_GOOGLE_SCOPES)
        except DefaultCredentialsError as e:
            raise LLMConfigError(
                "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or run "
                f"'gcloud auth application-default login'. Error: {e}"
            )

    project_id = _project_in_env() or project_in_auth
    if not project_id:
        raise _project_not_found()

    _google_credentials_cache = (credentials, project_id)
    return _google_credentials_cache


def get_vertex_config() -> dict:
    """
    Vertex AI の宛先 {"project_id", "location"} を返す。

    認証そのものは SDK が ADC（GOOGLE_APPLICATION_CREDENTIALS を含む）で行う。
    location は VERTEX_LOCATION > GOOGLE_CLOUD_LOCATION > "global"。
    """
    path = _service_account_path()
    project_id = _project_in_env() or (path and _project_in_file(path)) or _project_in_adc()
    if not project_id:
        raise _project_not_found()

    location = (
        os.environ.get("VERTEX_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "global"
    )
    return {"project_id": project_id, "location": location}


def _project_in_adc() -> str | None:
    try:
        import google.auth

        return google.auth.default()[1]
    except Exception:
        return None


# =============================================================================
# モデル・料金取得
# =============================================================================
def get_model(provider: str, model: Optional[str] = None) -> str:
    """モデル名を取得（指定がなければデフォルト）"""
    from .config import DEFAULT_MODELS

    if model:
        return model

    default = DEFAULT_MODELS.get(provider)
    if not default:
        raise LLMConfigError(f"No default model for provider: {provider}")

    return default


_unpriced: set[tuple[str, str]] = set()


def get_pricing(provider: str, model: str) -> dict:
    """料金（USD / 100 万トークン）を返す。料金表に無いモデルは 0 として、1 度だけ警告する。"""
    from .config import PRICING

    # vertexはgoogleと同じ料金体系
    pricing = PRICING.get("google" if provider == "vertex" else provider, {}).get(model)
    if pricing is not None:
        return pricing
    if (provider, model) not in _unpriced:
        _unpriced.add((provider, model))
        logger.warning(
            "料金表に %s / %s が無いので 0 ドルとして数えます（llm_kit/config.py の PRICING に足す）",
            provider, model,
        )
    return {"input": 0, "output": 0}


# =============================================================================
# ソフトバリデーション（修復パース）
# =============================================================================


def try_lenient_parse(raw_content: str, schema: type[BaseModel]) -> BaseModel | None:
    """
    raw出力をソフトバリデーションで修復パースする。
    修復できなければNoneを返す。

    修復対象:
    - markdownコードブロック除去
    - JSON前後のテキスト除去
    - 途中で切れた JSON（閉じていない文字列・配列・オブジェクト）
    - camelCase → snake_case フィールド名変換
    - 型の軽微な不一致（文字列→リスト、空文字→None等）
    - 余分なフィールドの無視
    """
    json_str = extract_json(raw_content)
    if json_str is None:
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    data = _normalize_field_names(data, schema)
    data = _coerce_field_types(data, schema)

    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


def extract_json(text: str) -> str | None:
    """テキストから最初の JSON オブジェクトを取り出す。見つからなければ None。

    LLMがよくやる次の崩れを吸収する:
    - ```json ... ``` / ``` ... ``` のコードフェンス（閉じ ``` が無くても可）
    - JSON の前後に付く前置き・説明などの地の文
    - max_tokens 等で末尾が切れた JSON → 開いたままの文字列・配列・オブジェクトを閉じて救済
    """
    if not text:
        return None
    text = text.strip()

    # コードフェンス除去（```json / ``` 開始。閉じフェンスが無くても中身を採用）
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)(?:```|\Z)", text, re.DOTALL)
    if fence and "{" in fence.group(1):
        text = fence.group(1).strip()

    # 最初の '{' から、対応する '}' までを取り出す（後続の地の文は無視）
    start = text.find("{")
    if start == -1:
        return None
    closers: list[str] = []  # まだ閉じていない括弧の、閉じる側の文字
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif closers and ch == closers[-1]:
            closers.pop()
            if not closers:
                return text[start:i + 1]

    # 途中で切れた → 開いた文字列と括弧を閉じて救済
    repaired = text[start:] + ('"' if in_str else "")
    return repaired + "".join(reversed(closers))


# =============================================================================
# フィードバック付きリトライメッセージ構築
# =============================================================================


def build_retry_messages(
    original_messages: list[BaseMessage],
    raw_output: str,
    error_detail: str,
    schema: type[BaseModel],
) -> list[BaseMessage]:
    """
    前回の出力とエラー内容をフィードバックとして追加したメッセージを構築する。

    Args:
        original_messages: 元のメッセージリスト（SystemMessage + HumanMessage）
        raw_output: 前回のLLM出力（raw）
        error_detail: パースエラーの詳細
        schema: 期待するPydanticスキーマ

    Returns:
        フィードバックメッセージを末尾に追加した新しいメッセージリスト
    """
    # raw_outputが長すぎる場合は切り詰め
    truncated_output = raw_output[:1000] if raw_output else "(空の出力)"

    feedback = (
        f"あなたの前回の出力は正しいJSON形式ではありませんでした。\n\n"
        f"【前回の出力】\n{truncated_output}\n\n"
        f"【エラー内容】\n{error_detail}\n\n"
        f"【期待するスキーマ】\n{_format_schema_description(schema)}\n\n"
        f"上記のスキーマに厳密に従ったJSON形式で再出力してください。"
        f"JSONのみを出力し、説明文やコードブロック記法は不要です。"
    )

    return [*original_messages, HumanMessage(content=feedback)]


def json_output_instruction(schema: type[BaseModel]) -> str:
    """初回プロンプト末尾に付与する、スキーマ厳守のJSON出力指示を返す。

    ainvoke_structured_with_retry は初回に schema を LLM へ渡さないため、
    これを付けないとモデルが地の文で答えてパース失敗→リトライになりやすい。
    """
    return (
        "\n\n---\n【出力形式（厳守）】\n"
        "次のフィールドだけを持つ JSON オブジェクトを 1 個だけ出力してください:\n"
        f"{_format_schema_description(schema)}\n"
        "JSON 以外（前置き・説明・理由・```コードブロック記法など）は一切出力しないこと。"
    )


# =============================================================================
# 内部ヘルパー
# =============================================================================


def _normalize_field_names(
    data: dict[str, Any], schema: type[BaseModel]
) -> dict[str, Any]:
    """camelCase等のフィールド名をスキーマのsnake_caseに正規化する"""
    schema_fields = set(schema.model_fields.keys())

    # 既にスキーマと一致するフィールドはそのまま
    if schema_fields <= set(data.keys()):
        return data

    # lowerCamelCase / UpperCamelCase → snake_case の対応表
    camel_to_snake: dict[str, str] = {}
    for field_name in schema_fields:
        parts = field_name.split("_")
        camel_to_snake[parts[0] + "".join(p.capitalize() for p in parts[1:])] = field_name
        camel_to_snake["".join(p.capitalize() for p in parts)] = field_name

    # 余分なフィールドは捨てる
    normalized = {}
    for key, value in data.items():
        if key in schema_fields:
            normalized[key] = value
        elif key in camel_to_snake:
            normalized[camel_to_snake[key]] = value
    return normalized


def _coerce_field_types(
    data: dict[str, Any], schema: type[BaseModel]
) -> dict[str, Any]:
    """スキーマの型ヒントに合わせて軽微な型変換を行う"""
    coerced = dict(data)

    for field_name, field_info in schema.model_fields.items():
        if field_name not in coerced:
            continue

        value = coerced[field_name]
        annotation = field_info.annotation

        # Optional[X] と X | None から X を取り出す
        args = get_args(annotation)
        is_optional = get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in args
        inner_type = annotation
        if is_optional:
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                inner_type = non_none_args[0]

        # 空文字 → None（Optional フィールドの場合）
        if is_optional and isinstance(value, str) and value.strip() == "":
            coerced[field_name] = None
            continue

        # 文字列 → リスト変換（List[str] を期待している場合）。カンマ・読点・改行で割る。
        if get_origin(inner_type) is list and isinstance(value, str):
            items = re.split(r"[,、\n]", value)
            coerced[field_name] = [item.strip() for item in items if item.strip()]

    return coerced


def _format_schema_description(schema: type[BaseModel]) -> str:
    """Pydanticスキーマを人間可読な説明文に変換する"""
    lines = []
    for field_name, field_info in schema.model_fields.items():
        desc = field_info.description or ""
        req_label = "必須" if field_info.is_required() else "任意"
        lines.append(f"- {field_name} ({field_info.annotation}, {req_label}): {desc}")
    return "\n".join(lines)
