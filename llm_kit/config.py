"""
LLM設定（定数のみ）
"""

# =============================================================================
# 料金表 (USD per 1M tokens)
# =============================================================================
PRICING = {
    "openai": {
        # ---- GPT-5.6 世代（2026-07-09 GA、コンテキスト 1.05M）----
        # 3 段構成。Sol=フラッグシップ / Terra=標準 / Luna=低コスト。
        # 272K トークンを超える入力は長コンテキスト扱いで、入力 2倍・出力 1.5倍。
        # Sol: 2026-08-21 に $5.00/$30.00 → $4.00/$20.00。
        #      「少なくとも 2026-11-21 まで」の販促価格。期限後は要再確認。
        "gpt-5.6-sol": {"input": 4.00, "output": 20.00},    # [PROMO ~2026-11-21] 通常 5.00 / 30.00
        "gpt-5.6": {"input": 4.00, "output": 20.00},        # sol のエイリアス
        "gpt-5.6-terra": {"input": 2.00, "output": 12.00},  # 2026-07-30 に 20% 値下げ
        "gpt-5.6-luna": {"input": 0.20, "output": 1.20},    # 2026-07-30 に 80% 値下げ
        # 限定アクセス（Trusted Access for Cyber）。単一ソースなので要確認。
        "gpt-5.6-cyber": {"input": 12.50, "output": 75.00},  # 要確認
        "gpt-5.5-cyber": {"input": 12.50, "output": 75.00},  # 要確認

        # ---- GPT-5.5 / 5.4 世代（現行の価格表に残っている）----
        "gpt-5.5": {"input": 5.00, "output": 30.00},
        "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
        "gpt-5.4": {"input": 2.50, "output": 15.00},
        "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
        "gpt-5.4-nano": {"input": 0.20, "output": 1.25},

        # ---- GPT-5.3 / 5.2 / 5.1 / 5（旧世代。5.2-pro は deprecated）----
        "gpt-5.3": {"input": 1.75, "output": 14.00},
        "gpt-5.3-codex": {"input": 1.75, "output": 14.00},
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-5.2-codex": {"input": 1.75, "output": 14.00},
        "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-5.1-codex": {"input": 1.25, "output": 10.00},
        "gpt-5.1-codex-max": {"input": 1.25, "output": 10.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5-pro": {"input": 15.00, "output": 120.00},

        # ---- GPT-4 系および Realtime ----
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-realtime-2": {"input": 4.00, "output": 24.00},
        "gpt-realtime-1.5": {"input": 4.00, "output": 16.00},
        "gpt-realtime": {"input": 4.00, "output": 16.00},
        "gpt-realtime-mini": {"input": 0.60, "output": 2.40},

        # ---- o シリーズ（レガシー）----
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o1-pro": {"input": 150.00, "output": 600.00},
        "o3": {"input": 2.00, "output": 8.00},
        "o3-pro": {"input": 20.00, "output": 80.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o4-mini": {"input": 1.10, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "output": 8.00},
    },
    "anthropic": {
        # 公式表と突き合わせ済み（2026-08-26 版）。
        # キャッシュは 5分書き込み 1.25倍 / 1時間書き込み 2倍 / 読み出し 0.1倍。
        # Claude 4.6 以降は 1M コンテキストまで同一レート（長コンテキスト割増なし）。
        # inference_geo="us" を指定すると全カテゴリ 1.1倍。

        # ---- Claude 5 世代 ----
        "claude-fable-5": {"input": 10.00, "output": 50.00},
        "claude-mythos-5": {"input": 10.00, "output": 50.00},  # Project Glasswing 限定
        "claude-opus-5": {"input": 5.00, "output": 25.00},
        # ★変更: sonnet-5 は $2/$10 が「正式な標準価格」になった。
        #   2026-09-01 に $3/$15 へ上がる予定は撤回された（公式表の注記）。
        "claude-sonnet-5": {"input": 2.00, "output": 10.00},

        # Fast mode（research preview、Opus 5 / 4.8 のみ）は別レート。
        # 使っているなら以下のキーで分けるとよい。
        # "claude-opus-5:fast": {"input": 10.00, "output": 50.00},
        # "claude-opus-4-8:fast": {"input": 10.00, "output": 50.00},

        # ---- Claude 4.8 / 4.7 / 4.6 世代 ----
        "claude-opus-4-8": {"input": 5.00, "output": 25.00},
        "claude-opus-4-7": {"input": 5.00, "output": 25.00},
        "claude-opus-4-6": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},

        # ---- Claude 4.5 シリーズ ----
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},

        # ---- 退役済み（Bedrock / Google Cloud には残る。過去ログの再計算用）----
        "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},  # 公式表から削除済み
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},  # 公式表から削除済み
    },
    "google": {
        # 公式の Gemini Developer API 価格表と突き合わせ済み（2026-08-26 版）。
        # Flash 系の値上げが 2027-01-01 に予定されている点に注意。

        # ---- Gemini 3.7 / 3.6 Flash ----
        # 3.7 Flash は 2026-08-13 GA。導入価格 $0.75/$3.75 が 3.6 Flash にも適用された。
        # 2027-01-01 から両方とも $1.50/$7.50 に戻る。
        "gemini-3.7-flash": {"input": 0.75, "output": 3.75},  # [PROMO ~2026-12-31] 通常 1.50 / 7.50
        "gemini-3.6-flash": {"input": 0.75, "output": 3.75},  # [PROMO ~2026-12-31] 通常 1.50 / 7.50

        # ---- Gemini 3.5 / 3.1 世代 ----
        "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
        "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
        # Pro は 200K トークン超で input 4.00 / output 18.00 に上がる。
        "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
        "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},  # 音声入力は 0.50

        # ---- Gemini 3 世代 ----
        "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},  # 音声入力は 1.00
        # gemini-3-pro-preview は公式の価格表から消えた（3.1-pro-preview に統合）。
        # 過去ログ用に残すなら以下。新規では 3.1 を使うこと。
        "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},

        # ---- Gemini 2.5 世代（2026-10-16 に退役予定）----
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # 200K 超で 2.50 / 15.00
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},  # 音声入力は 1.00
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},  # 音声入力は 0.30

        # ---- Gemini 2.0 世代: 2026-06-01 に停止済み。過去ログ用にのみ残す ----
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    },
    "bedrock": {
        # ★重要: Sonnet 4.5 以降、GEO プロファイル（apac. / jp. / us. / eu.）は
        #   global. より 10% 高い。以前の「Anthropic 直と同額」は 4.5 世代から成り立たない。
        #   Sonnet 4 以前のモデルは従来どおり据え置き。

        # GEO CRIS（10% 割増）
        "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.30, "output": 16.50},
        "apac.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.10, "output": 5.50},
        # Sonnet 4 以前 = 割増なし
        "apac.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.00, "output": 15.00},
        "apac.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 3.00, "output": 15.00},
        "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 3.00, "output": 15.00},

        # Global CRIS（Anthropic 直と同額。データ所在の制約がなければこちらが安い）
        # Claude 5 世代は Bedrock では版番号なしの ID。Opus 5 に jp. プロファイルはなく、
        # 日本リージョンからは global. 経由でのみ使える。
        "global.anthropic.claude-fable-5": {"input": 10.00, "output": 50.00},
        "global.anthropic.claude-opus-5": {"input": 5.00, "output": 25.00},
        "global.anthropic.claude-sonnet-5": {"input": 2.00, "output": 10.00},
        "global.anthropic.claude-opus-4-6-v1": {"input": 5.00, "output": 25.00},
        "global.anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "global.anthropic.claude-opus-4-5-20251101-v1:0": {"input": 5.00, "output": 25.00},
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.00},
        "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.00, "output": 5.00},
    },
}


# =============================================================================
# デフォルトモデル
# =============================================================================
DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "anthropic": "claude-sonnet-5",
    "google": "gemini-3-flash-preview",
    "vertex": "gemini-3-flash-preview",
    "claude_code": "claude-sonnet-5",
    # 東京(ap-northeast-1)ネイティブの on-demand モデル。
    # Claude 4.x を使う場合は apac.* クロスリージョン推論プロファイルが必要（東京primary）。
    "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}


# =============================================================================
# デフォルトエンベディングモデル
# =============================================================================
DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "google": "models/text-embedding-004",
    "vertex": "text-embedding-005",
}


# =============================================================================
# エンベディング料金表 (USD per 1M tokens)
# =============================================================================
EMBEDDING_PRICING = {
    "openai": {
        "text-embedding-3-small": {"input": 0.02},
        "text-embedding-3-large": {"input": 0.13},
        "text-embedding-ada-002": {"input": 0.10},
    },
    "google": {
        "models/text-embedding-004": {"input": 0.00},  # 無料枠あり
        "text-embedding-004": {"input": 0.00},
        "text-embedding-005": {"input": 0.00},
    },
}


# =============================================================================
# デフォルトプロバイダー
# =============================================================================
DEFAULT_PROVIDER = "vertex"


# =============================================================================
# リランク設定
# =============================================================================
DEFAULT_RERANK_PROVIDER = "bedrock"

DEFAULT_RERANK_MODELS = {
    # Bedrock Rerank API（SigV4認証。bearerキー不可）。東京で利用可。
    "bedrock": "cohere.rerank-v3-5:0",
    # Cohere 直API（COHERE_API_KEY）
    "cohere": "rerank-v3.5",
}


# =============================================================================
# 為替レート（概算）
# =============================================================================
USD_TO_JPY = 150.0
