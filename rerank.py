"""
リランク（関連度による並べ替え）

汎用インターフェース。プロバイダーを切り替えて利用する:
    - bedrock: cohere.rerank-v3-5:0 等を Bedrock Rerank API 経由（SigV4認証）
    - cohere : Cohere 直API（COHERE_API_KEY）

Bedrock 経路は bearer API キーではなく通常の AWS 認証情報（SigV4）が必要。
aws-vault を使うなら、プロファイルを指定して実行する:
    aws-vault exec <profile> -- uv run python your_script.py

Example:
    from llm_kit import rerank

    results = rerank(
        "ログイン制限を解除したい",
        [
            "パスワード再設定の手順",
            "アカウントのログイン制限（ロック）の解除方法",
            "退会手続きについて",
        ],
        top_n=2,
    )
    for r in results:
        print(r["score"], r["document"])
"""

from typing import Any, Optional

from .config import DEFAULT_RERANK_PROVIDER, DEFAULT_RERANK_MODELS
from .exceptions import LLMConfigError

# boto3 クライアントはスレッドセーフなので使い回す（接続確立コスト削減）
_bedrock_clients: dict[str, Any] = {}


def _get_bedrock_client(region: str):
    client = _bedrock_clients.get(region)
    if client is None:
        import boto3

        client = boto3.client("bedrock-agent-runtime", region_name=region)
        _bedrock_clients[region] = client
    return client


def _extract_texts(documents: list, text_key: Optional[str]) -> list[str]:
    """documents が dict のリストなら text_key で本文を取り出す"""
    texts = []
    for doc in documents:
        if isinstance(doc, str):
            texts.append(doc)
        elif isinstance(doc, dict):
            if not text_key:
                raise LLMConfigError(
                    "documents が dict の場合は text_key を指定してください"
                )
            texts.append(doc[text_key])
        else:
            raise LLMConfigError(f"未対応の document 型: {type(doc)}")
    return texts


def rerank(
    query: str,
    documents: list,
    top_n: Optional[int] = None,
    provider: str = None,
    model: str = None,
    region: str = None,
    text_key: Optional[str] = None,
    return_documents: bool = True,
) -> list[dict]:
    """
    クエリとの関連度で documents を並べ替える

    Args:
        query: 検索クエリ
        documents: 並べ替え対象。str のリスト、または dict のリスト（text_key で本文指定）
        top_n: 返す上位件数（None なら全件）
        provider: "bedrock" | "cohere"（省略時 DEFAULT_RERANK_PROVIDER）
        model: モデルID（省略時プロバイダー既定）
            - bedrock: "cohere.rerank-v3-5:0" / "amazon.rerank-v1:0" 等
            - cohere : "rerank-v3.5" 等
        region: AWSリージョン（bedrockのみ。省略時 AWS_REGION or ap-northeast-1）
        text_key: documents が dict の場合の本文キー
        return_documents: True なら結果に元 document を含める

    Returns:
        スコア降順の list[dict]:
            {"index": 元インデックス, "score": 関連度, "document": 元document(任意)}

    Example:
        rerank("東京の天気", ["大阪の飲食店", "東京の週間予報"], top_n=1)
        # -> [{"index": 1, "score": 0.98, "document": "東京の週間予報"}]
    """
    provider = provider or DEFAULT_RERANK_PROVIDER
    model = model or DEFAULT_RERANK_MODELS.get(provider)
    if not model:
        raise LLMConfigError(f"No default rerank model for provider: {provider}")

    if not documents:
        return []

    texts = _extract_texts(documents, text_key)
    n = top_n if top_n is not None else len(texts)

    if provider == "bedrock":
        import os

        region = region or os.environ.get("AWS_REGION", "ap-northeast-1")
        client = _get_bedrock_client(region)

        # 短縮モデルIDは foundation-model ARN に展開
        model_arn = (
            model
            if model.startswith("arn:")
            else f"arn:aws:bedrock:{region}::foundation-model/{model}"
        )

        resp = client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": t},
                    },
                }
                for t in texts
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": model_arn},
                    "numberOfResults": n,
                },
            },
        )
        ranked = [
            {"index": r["index"], "score": r["relevanceScore"]}
            for r in resp["results"]
        ]

    elif provider == "cohere":
        import cohere

        co = cohere.Client()  # COHERE_API_KEY を環境変数から取得
        resp = co.rerank(query=query, documents=texts, model=model, top_n=n)
        ranked = [
            {"index": r.index, "score": r.relevance_score} for r in resp.results
        ]

    else:
        raise LLMConfigError(f"Unknown rerank provider: {provider}")

    if return_documents:
        for r in ranked:
            r["document"] = documents[r["index"]]
    return ranked
