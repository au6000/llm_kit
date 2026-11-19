"""
ユーティリティ関数（認証、モデル取得、JSONパース、Structured Output のソフトバリデーション）
"""

import os
import json
import re
from typing import Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from .exceptions import LLMConfigError

# .envファイルを自動読み込み
load_dotenv()


# =============================================================================
# JSONパース
# =============================================================================
def extract_json(content: str) -> str:
    """
    レスポンスからJSONを抽出（マークダウンコードブロックを除去）

    対応フォーマット:
    - 純粋なJSON: {"key": "value"}
    - コードブロック: ```json\n{"key": "value"}\n```
    - コードブロック(言語指定なし): ```\n{"key": "value"}\n```
    """
    content = content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]  # 最初の行を除去
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    return content.strip()


def parse_json(content: str) -> dict:
    """
    LLM出力からJSONをパースする

    Args:
        content: LLMの出力テキスト

    Returns:
        パースされた辞書

    Raises:
        json.JSONDecodeError: JSONのパースに失敗した場合
    """
    return json.loads(extract_json(content))


# =============================================================================
# API KEY取得
# =============================================================================
def get_api_key(provider: str) -> str:
    """
    環境変数からAPI KEYを取得

    環境変数名:
        - openai: OPENAI_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - google: GOOGLE_API_KEY
        - vertex: 不要（サービスアカウントJSONで認証）
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    # Vertexはサービスアカウント認証のためAPI KEYは不要
    if provider == "vertex":
        return ""

    env_var = env_vars.get(provider)
    if not env_var:
        raise LLMConfigError(f"Unknown provider: {provider}")

    api_key = os.environ.get(env_var)
    if not api_key:
        raise LLMConfigError(
            f"API KEY not found. Set environment variable: {env_var}"
        )

    return api_key


_google_credentials_cache: tuple | None = None


def get_google_credentials():
    """
    Google GenAI用のサービスアカウント認証情報を取得（キャッシュ付き）

    初回呼び出し時に認証情報を取得し、以降はキャッシュを返す。
    並列実行時の大量認証リクエストを防止する。

    認証方式（優先順位）:
        1. GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス
        2. Application Default Credentials (ADC): gcloud auth application-default login

    環境変数:
        - GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス（任意）
        - VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT: GCPプロジェクトID

    Returns:
        tuple: (credentials, project_id)
    """
    global _google_credentials_cache

    if _google_credentials_cache is not None:
        return _google_credentials_cache

    from google.oauth2 import service_account
    import google.auth

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    credentials = None
    project_id = None

    if credentials_path:
        # サービスアカウントJSONファイルから認証
        if not os.path.exists(credentials_path):
            raise LLMConfigError(
                f"Service account JSON file not found: {credentials_path}"
            )

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # プロジェクトIDをJSONから取得
        try:
            with open(credentials_path, "r") as f:
                creds_data = json.load(f)
                project_id = creds_data.get("project_id")
        except (json.JSONDecodeError, IOError):
            pass
    else:
        # Application Default Credentials (ADC) を使用
        try:
            credentials, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except google.auth.exceptions.DefaultCredentialsError as e:
            raise LLMConfigError(
                f"Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'. Error: {e}"
            )

    # 環境変数からプロジェクトIDを上書き
    project_id = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or project_id

    if not project_id:
        raise LLMConfigError(
            "Project ID not found. Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable."
        )

    _google_credentials_cache = (credentials, project_id)
    return _google_credentials_cache


def get_vertex_config() -> dict:
    """
    Vertex AI用の設定を取得

    認証方式（優先順位）:
        1. GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス
        2. Application Default Credentials (ADC): gcloud auth application-default login

    環境変数:
        - GOOGLE_APPLICATION_CREDENTIALS: サービスアカウントJSONファイルのパス（任意）
        - VERTEX_PROJECT_ID: GCPプロジェクトID（必須、または GOOGLE_CLOUD_PROJECT）
        - VERTEX_LOCATION / GOOGLE_CLOUD_LOCATION: リージョン（デフォルト: global）
    """
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    auth_method = None

    # 認証方式1: サービスアカウントJSONファイル
    if credentials_path:
        if not os.path.exists(credentials_path):
            raise LLMConfigError(
                f"Service account JSON file not found: {credentials_path}"
            )
        auth_method = "service_account"
    else:
        # 認証方式2: Application Default Credentials (ADC)
        auth_method = "adc"

    # プロジェクトIDの取得（優先順位: VERTEX_PROJECT_ID > GOOGLE_CLOUD_PROJECT > JSONから取得）
    project_id = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        if credentials_path:
            # サービスアカウントJSONからプロジェクトIDを取得
            try:
                with open(credentials_path, "r") as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get("project_id")
            except (json.JSONDecodeError, IOError) as e:
                raise LLMConfigError(f"Failed to read service account JSON: {e}")

        if not project_id:
            # ADC使用時はgcloudの設定からproject_idを取得試行
            try:
                import google.auth
                _, adc_project = google.auth.default()
                project_id = adc_project
            except Exception:
                pass

    if not project_id:
        raise LLMConfigError(
            "VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT not found. "
            "Set the environment variable or ensure gcloud is configured with a default project."
        )

    # VERTEX_LOCATION が無ければ、Google の SDK と同じ名前の GOOGLE_CLOUD_LOCATION を使う。
    location = (
        os.environ.get("VERTEX_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "global"
    )

    return {
        "credentials_path": credentials_path,  # Noneの場合はADCを使用
        "project_id": project_id,
        "location": location,
        "auth_method": auth_method,
    }


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


def get_pricing(provider: str, model: str) -> dict:
    """料金情報を取得"""
    from .config import PRICING

    # vertexはgoogleと同じ料金体系
    pricing_key = "google" if provider == "vertex" else provider
    provider_pricing = PRICING.get(pricing_key, {})
    return provider_pricing.get(model, {"input": 0, "output": 0})


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
    - camelCase → snake_case フィールド名変換
    - 型の軽微な不一致（文字列→リスト、空文字→None等）
    - 余分なフィールドの無視
    """
    if not raw_content or not raw_content.strip():
        return None

    # Step 1: JSON文字列を抽出
    json_str = _extract_json(raw_content)
    if json_str is None:
        return None

    # Step 2: JSONパース
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Step 3: フィールド名の正規化
    data = _normalize_field_names(data, schema)

    # Step 4: 型の修復
    data = _coerce_field_types(data, schema)

    # Step 5: Pydanticで再パース
    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


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
    schema_description = _format_schema_description(schema)

    # raw_outputが長すぎる場合は切り詰め
    truncated_output = raw_output[:1000] if raw_output else "(空の出力)"

    feedback = (
        f"あなたの前回の出力は正しいJSON形式ではありませんでした。\n\n"
        f"【前回の出力】\n{truncated_output}\n\n"
        f"【エラー内容】\n{error_detail}\n\n"
        f"【期待するスキーマ】\n{schema_description}\n\n"
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


def _extract_json(text: str) -> str | None:
    """テキストからJSONオブジェクト部分を抽出する。

    LLMがよくやる次の崩れを吸収する:
    - ```json ... ``` / ``` ... ``` のコードフェンス（閉じ ``` が無くても可）
    - JSON の前後に付く前置き・説明などの地の文
    - max_tokens 等で末尾が切れた（閉じ括弧不足の）JSON → 不足分を補って救済
    """
    if not text:
        return None
    text = text.strip()

    # コードフェンス除去（```json / ``` 開始。閉じフェンスが無くても中身を採用）
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)(?:```|\Z)", text, re.DOTALL)
    if fence and "{" in fence.group(1):
        text = fence.group(1).strip()

    # 最初の '{' から括弧カウントで対応する '}' までを取り出す（後続の地の文は無視）
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # 閉じ括弧不足（途中切れ）→ 開いた文字列/括弧を補完して救済
    if depth > 0:
        repaired = text[start:]
        if in_str:
            repaired += '"'
        return repaired + "}" * depth
    return None


def _normalize_field_names(
    data: dict[str, Any], schema: type[BaseModel]
) -> dict[str, Any]:
    """camelCase等のフィールド名をスキーマのsnake_caseに正規化する"""
    schema_fields = set(schema.model_fields.keys())

    # 既にスキーマと一致するフィールドはそのまま
    if schema_fields <= set(data.keys()):
        return data

    # camelCase → snake_case 変換マッピングを作成
    camel_to_snake: dict[str, str] = {}
    for field_name in schema_fields:
        # snake_case → camelCase の変換パターンを生成
        parts = field_name.split("_")
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        camel_to_snake[camel] = field_name
        # lowerCamelCase と UpperCamelCase 両対応
        upper_camel = "".join(p.capitalize() for p in parts)
        camel_to_snake[upper_camel] = field_name

    normalized = {}
    for key, value in data.items():
        if key in schema_fields:
            normalized[key] = value
        elif key in camel_to_snake:
            normalized[camel_to_snake[key]] = value
        else:
            # 余分なフィールドは無視（捨てる）
            pass

    return normalized


def _coerce_field_types(
    data: dict[str, Any], schema: type[BaseModel]
) -> dict[str, Any]:
    """スキーマの型ヒントに合わせて軽微な型変換を行う"""
    import typing
    from typing import get_args, get_origin

    coerced = dict(data)

    for field_name, field_info in schema.model_fields.items():
        if field_name not in coerced:
            continue

        value = coerced[field_name]
        annotation = field_info.annotation

        # Optional[X] → X を取り出す
        origin = get_origin(annotation)
        args = get_args(annotation)
        inner_type = annotation
        is_optional = False

        if origin is typing.Union and type(None) in args:
            is_optional = True
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                inner_type = non_none_args[0]

        inner_origin = get_origin(inner_type)

        # 空文字 → None（Optional フィールドの場合）
        if is_optional and isinstance(value, str) and value.strip() == "":
            coerced[field_name] = None
            continue

        # 文字列 → リスト変換（List[str] を期待している場合）
        if inner_origin is list and isinstance(value, str):
            # カンマ or 改行区切りでsplit
            items = re.split(r"[,、\n]", value)
            coerced[field_name] = [item.strip() for item in items if item.strip()]
            continue

    return coerced


def _format_schema_description(schema: type[BaseModel]) -> str:
    """Pydanticスキーマを人間可読な説明文に変換する"""
    lines = []
    for field_name, field_info in schema.model_fields.items():
        desc = field_info.description or ""
        annotation = field_info.annotation
        required = field_info.is_required()
        req_label = "必須" if required else "任意"
        lines.append(f"- {field_name} ({annotation}, {req_label}): {desc}")
    return "\n".join(lines)
