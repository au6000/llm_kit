"""
LLM設定（定数のみ）

料金と既定のモデルは 2026-11-19 に次の公式の値と突き合わせた:
  OpenAI   https://developers.openai.com/api/docs/pricing
  Anthropic https://platform.claude.com/docs/en/about-claude/pricing と models/overview
  Google   https://aistudio.google.com/docs/pricing（Gemini Developer API）
  Gateway  https://ai-gateway.vercel.sh/v1/models（Vercel AI Gateway の公開のモデル一覧。鍵は要らない）
Bedrock の料金ページは中身を読めなかったので、Bedrock の値は Anthropic の料金ページの
「global は直と同額、地域のエンドポイントは 10% 増し」から出している。
"""

# =============================================================================
# 料金表 (USD per 1M tokens)
# =============================================================================
#
# 1 つのモデルに 1 つの値しか持てない。長いプロンプトの割増や、日付つきの値上げは
# 注記にだけ書いてあり、集計（usage_tracker）は表の値で数える。
# 表に無いモデルは 0 ドルとして数え、警告を 1 度出す（utils.get_pricing）。

PRICING = {
    "openai": {
        # 272K トークンを超える入力は長いコンテキスト扱いで、GPT-5.4 以降（mini / nano を除く）は
        # 入力 2 倍・出力 1.5 倍。

        # ---- GPT-6 世代 ----
        "gpt-6.2-astra": {"input": 8.00, "output": 40.00},   # 2026-11-17
        "gpt-6.2-sol": {"input": 2.00, "output": 10.00},     # 2026-11-17
        "gpt-6.1-sol": {"input": 2.00, "output": 10.00},
        "gpt-6.1-terra": {"input": 1.00, "output": 5.00},
        "gpt-6-astra": {"input": 10.00, "output": 50.00},
        "gpt-6-sol": {"input": 2.00, "output": 10.00},
        "gpt-6-luna": {"input": 0.10, "output": 0.50},

        # ---- GPT-5.6 世代（2026-07-09 GA）----
        # Sol は 2026-11-30 までの販促価格。期限後の値は公式の表に書いていない（以前の通常は 5.00 / 30.00）。
        "gpt-5.6-sol": {"input": 4.00, "output": 20.00},    # [PROMO ~2026-11-30]
        "gpt-5.6": {"input": 4.00, "output": 20.00},        # sol のエイリアス
        "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
        "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
        # 限定アクセス（Trusted Access for Cyber）。単一ソースなので要確認。
        "gpt-5.6-cyber": {"input": 12.50, "output": 75.00},  # 要確認
        "gpt-5.5-cyber": {"input": 12.50, "output": 75.00},  # 要確認

        # ---- GPT-5.5 / 5.4 世代 ----
        "gpt-5.5": {"input": 5.00, "output": 30.00},
        "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
        "gpt-5.4": {"input": 2.50, "output": 15.00},
        "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
        "gpt-5.4-nano": {"input": 0.20, "output": 1.25},

        # ---- GPT-5.3 / 5.2 / 5.1 / 5（旧世代。過去ログの再計算用）----
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
        # キャッシュは 5分書き込み 1.25倍 / 1時間書き込み 2倍 / 読み出し 0.1倍。
        # ただし読み出しは Fable 5.2・5.1 が 0.025倍、Opus 5.5・Sonnet 5.5 が 0.05倍。
        # Claude 4.6 以降は inference_geo="us" で全カテゴリ 1.1倍。

        # ---- Claude 5.5 / 5.2 / 5.1 世代 ----
        "claude-fable-5-2": {"input": 8.00, "output": 40.00},
        "claude-mythos-5-2": {"input": 8.00, "output": 40.00},  # 限定アクセス
        "claude-fable-5-1": {"input": 10.00, "output": 50.00},
        "claude-mythos-5-1": {"input": 10.00, "output": 50.00},  # 限定アクセス
        "claude-opus-5-5": {"input": 4.00, "output": 20.00},
        "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
        # Haiku 5.5 は 100,000 トークンを超えるプロンプトで 0.50 / 2.50 に上がる。
        "claude-haiku-5-5": {"input": 0.10, "output": 0.50},

        # ---- Claude 5 世代 ----
        "claude-fable-5": {"input": 10.00, "output": 50.00},
        "claude-mythos-5": {"input": 10.00, "output": 50.00},  # 限定アクセス
        "claude-opus-5": {"input": 5.00, "output": 25.00},
        "claude-sonnet-5": {"input": 2.00, "output": 10.00},

        # Fast mode（Fable 5.2 / Opus 5.5 / Opus 5、Opus 4.8 は非推奨）は別レート。
        # 使っているなら以下のキーで分けるとよい。
        # "claude-fable-5-2:fast": {"input": 16.00, "output": 80.00},
        # "claude-opus-5-5:fast": {"input": 8.00, "output": 40.00},
        # "claude-opus-5:fast": {"input": 10.00, "output": 50.00},

        # ---- Claude 4.8 / 4.7 / 4.6 世代 ----
        "claude-opus-4-8": {"input": 5.00, "output": 25.00},
        "claude-opus-4-7": {"input": 5.00, "output": 25.00},
        "claude-opus-4-6": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},

        # ---- Claude 4.5 シリーズ（Sonnet 4.5 は非推奨）----
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},

        # ---- 退役済み（Bedrock / Google Cloud には残るものがある。過去ログの再計算用）----
        "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    "google": {
        # Gemini Developer API の有料の Standard の値。

        # ---- Gemini 4 世代 ----
        # carbon と argon は 2026-11-30 まで 2.00 / 10.00、2026-12-01 から 4.00 / 20.00。
        "gemini-4-carbon": {"input": 2.00, "output": 10.00},  # [~2026-11-30] 以後 4.00 / 20.00
        "gemini-4-argon": {"input": 2.00, "output": 10.00},   # [~2026-11-30] 以後 4.00 / 20.00
        "gemini-4-neon": {"input": 3.00, "output": 15.00},
        "gemini-4-hydrogen": {"input": 1.50, "output": 7.50},

        # ---- Gemini 3.8 / 3.7 / 3.6 Flash ----
        # 3.8 と 3.6 は 2027-01-01 から 1.50 / 7.50 に上がる。
        "gemini-3.8-flash": {"input": 0.75, "output": 3.75},  # [~2026-12-31] 以後 1.50 / 7.50
        "gemini-3.6-flash": {"input": 0.75, "output": 3.75},  # [~2026-12-31] 以後 1.50 / 7.50
        # 3.7 Flash は 2026-11-19 の公式の表から消えた（Gateway には同じ値で残る）。
        "gemini-3.7-flash": {"input": 0.75, "output": 3.75},

        # ---- Gemini 3.5 / 3.1 世代 ----
        # 3.5 Flash（lite でない方）も 2026-11-19 の公式の表から消えた（Gateway には同じ値で残る）。
        "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
        "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
        # Pro は 200K トークン超で input 4.00 / output 18.00 に上がる。
        "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
        "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},  # 音声入力は 0.50

        # ---- Gemini 3 世代 ----
        "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},  # 音声入力は 1.00
        # gemini-3-pro-preview は公式の表から消えた（3.1-pro-preview に統合）。過去ログ用。
        "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},

        # ---- Gemini 2.5 世代 ----
        # 2.5 Pro と 2.5 Flash は非推奨。2026-12-01 から、2.5 Pro は gemini-3.8-flash に、
        # 2.5 Flash は gemini-3.5-flash-lite に振り替えられ、その値で課金される。
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # 200K 超で 2.50 / 15.00
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},  # 音声入力は 1.00
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},  # 音声入力は 0.30

        # ---- Gemini 2.0 世代: 2026-06-01 に停止済み。過去ログ用にのみ残す ----
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    },
    "bedrock": {
        # Sonnet 4.5 以降は、地域のプロファイル（us. / eu. / jp. / apac.）が global. より 10% 高い。
        # Sonnet 4 以前は据え置き。
        # Fable 5.1 / 5、Opus 5.5 / 5、Sonnet 5.5 / 5、Haiku 5.5、Opus 4.8 / 4.7 は
        # bedrock-runtime では InvokeModel でだけ呼べる（Converse で呼べるかは確かめていない）。
        # get_llm("bedrock") は Converse を使うので、ここには Converse で呼べると確かめた ID だけを置く。

        # 日本だけを通るプロファイル（10% 増し）
        "jp.anthropic.claude-opus-4-6-v1": {"input": 5.50, "output": 27.50},
        "jp.anthropic.claude-sonnet-4-6": {"input": 3.30, "output": 16.50},
        "jp.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.30, "output": 16.50},  # 非推奨。EOL 2027-04-08

        # アジア太平洋のプロファイル（10% 増し。Sonnet 4 以前は割増なし）
        "apac.anthropic.claude-opus-4-6-v1": {"input": 5.50, "output": 27.50},
        # 次の 2 つは 2026-11-19 の Anthropic の表に apac. が無い（過去ログ用に残す）。
        "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.30, "output": 16.50},
        "apac.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.10, "output": 5.50},
        "apac.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.00, "output": 15.00},  # 2026-10-14 EOL
        "apac.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 3.00, "output": 15.00},  # 近隣の地域で 2026-11-13 EOL
        "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 3.00, "output": 15.00},  # 近隣の地域で 2026-11-13 EOL

        # 地域を固定しない基本のモデル ID（オンデマンド。過去ログ用）
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 3.00, "output": 15.00},

        # global（Anthropic 直と同額。データの置き場に制約がなければこちらが安い）
        "global.anthropic.claude-fable-5-2": {"input": 8.00, "output": 40.00},
        "us.anthropic.claude-fable-5-2": {"input": 8.80, "output": 44.00},  # jp. / apac. は無い
        "global.anthropic.claude-fable-5": {"input": 10.00, "output": 50.00},
        "global.anthropic.claude-opus-5": {"input": 5.00, "output": 25.00},
        "global.anthropic.claude-sonnet-5": {"input": 2.00, "output": 10.00},
        "global.anthropic.claude-opus-4-6-v1": {"input": 5.00, "output": 25.00},
        "global.anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "global.anthropic.claude-opus-4-5-20251101-v1:0": {"input": 5.00, "output": 25.00},
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.00},
        "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.00, "output": 5.00},
    },
    "gateway": {
        # Vercel AI Gateway の判定モデル（/v1/models で type が "evaluation" のもの）。
        # 状態と型つきの問い（boolean / choice / score）を渡すと確率を返す。入力トークンだけに
        # 課金され、出力は 0。キーは Gateway のモデル ID（"<提供元>/<名前>"）。
        # get_llm では呼べない（チャットではなく判定の API で呼ぶ）。記録するときは
        # add_usage("gateway", "typesafe-ai/jev", ...) のように、この名前で渡す。
        "typesafe-ai/jev": {"input": 0.042, "output": 0.0},
        "typesafe-ai/jev-1.14.0": {"input": 0.042, "output": 0.0},
        "typesafe-ai/jev-1.13.0": {"input": 0.042, "output": 0.0},
        "google/gemini-decision-1": {"input": 0.10, "output": 0.0},
        "openai/gpt-6-luna-decisions": {"input": 0.10, "output": 0.0},  # 272K 超で 0.20
        "microsoft/decision-1.1": {"input": 0.042, "output": 0.0},
        "microsoft/microsoft-decision-1": {"input": 0.042, "output": 0.0},
        "liquid/d1.2-ultra": {"input": 0.042, "output": 0.0},
        "liquid/d1.2": {"input": 0.02, "output": 0.0},
        "liquid/d1-ultra": {"input": 0.042, "output": 0.0},
        "liquid/d1": {"input": 0.02, "output": 0.0},
        "baseten/mammoth-1": {"input": 0.10, "output": 0.0},
        "baseten/elephant-1": {"input": 0.035, "output": 0.0},
        "baseten/joey-1": {"input": 0.01, "output": 0.0},
        "heygen/heygen-decision-1": {"input": 0.14, "output": 0.0},
        "heygen/heygen-decision-1-turbo": {"input": 0.09, "output": 0.0},
        "perplexity/pplx-decider-v1.3": {"input": 0.02, "output": 0.0},
        "perplexity/pplx-decider-v1.2": {"input": 0.02, "output": 0.0},
        "perplexity/pplx-decider-flash-v1.3": {"input": 0.01, "output": 0.0},
        "perplexity/pplx-decider-flash-v1.2": {"input": 0.01, "output": 0.0},
        "nace-ai/drex-v2.0": {"input": 0.03, "output": 0.0},
        "nace-ai/drex-max-v1.1": {"input": 0.03, "output": 0.0},
        "nace-ai/drex-max-finance": {"input": 0.03, "output": 0.0},
        "convaiinnovations/laya": {"input": 0.0105, "output": 0.0},
        "scaledown/scaledown-classify-1": {"input": 0.05, "output": 0.0},
        # 1 回ごとの課金（トークン課金ではない）。2026-11-19 時点で 0 ドル。
        "parallel/parallel-decisions-fast": {"input": 0.0, "output": 0.0},
    },
}


# =============================================================================
# デフォルトモデル
# =============================================================================
DEFAULT_MODELS = {
    "openai": "gpt-6.2-sol",
    "anthropic": "claude-sonnet-5-5",
    # 3.8 Flash（0.75 / 3.75、2027-01-01 から 1.50 / 7.50）より安いので、3 Flash Preview のまま。
    "google": "gemini-3-flash-preview",
    "vertex": "gemini-3-flash-preview",
    "claude_code": "claude-sonnet-5-5",
    # 日本の中だけを通る推論プロファイル（jp.、global. より 10% 高い）。以前の既定
    # anthropic.claude-3-5-sonnet-20240620-v1:0（東京ネイティブのオンデマンド）は、
    # 近隣の地域で 2026-11-13 に EOL になった。
    "bedrock": "jp.anthropic.claude-sonnet-4-6",
}


# =============================================================================
# デフォルトエンベディングモデル
# =============================================================================
DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    # text-embedding-004 は 2026-11-19 の公式の価格表に無い。使えなくなっていたら
    # models/gemini-embedding-001 に替える（ベクトルの次元が変わるので、作った索引は作り直しになる）。
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
        "models/gemini-embedding-2": {"input": 0.20},  # 文字の入力。画像・音声・動画は別の値
        "models/gemini-embedding-001": {"input": 0.15},
        "models/text-embedding-004": {"input": 0.00},  # 無料だった。2026-11-19 の公式の価格表には無い
    },
    "vertex": {
        "text-embedding-005": {"input": 0.025},  # Gateway の一覧の値
        "text-multilingual-embedding-002": {"input": 0.025},  # Gateway の一覧の値
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
# 為替レート（概算。2026-11-19 には見直していない）
# =============================================================================
USD_TO_JPY = 150.0
