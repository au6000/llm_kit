# llm_kit

LLM を呼ぶときに毎回要る機能を、ひととおり先に作っておいたライブラリ。
中身は LangChain の上に作ってあり、`get_llm()` が返すのは LangChain のチャットモデルなので、LangGraph にもそのまま渡せる。

## できること

- **プロバイダの切り替え**: `get_llm(provider, model)` の引数を変えるだけで、OpenAI・Anthropic・Google（Gemini API）・Vertex・Bedrock・Claude Code（CLI 経由）を切り替える
- **文字列で受ける**: `invoke` / `ainvoke` / `stream` / `astream`。返事がブロックの list で届くモデル（Gemini 3 など）でも、文字のブロックだけを繋いだ文字列を返す
- **構造化出力**: `invoke_structured` / `ainvoke_structured`。`ainvoke_structured_with_retry` は、返ってきた JSON が崩れていたらまず緩く読み直し（前後の地の文・途中で切れた JSON・camelCase の名前を直す）、それでも読めなければ誤りを添えて送り直す（既定で最大 3 回）
- **ストリーミング**: `stream` / `astream`、およびツール呼び出しの引数を届いた分から受け取る `astream_tool_calls`（OpenAI・Vertex）
- **同時に呼ぶ数の上限**: `configure_llm_concurrency`。`ainvoke` と `ainvoke_structured` 系に効く
- **利用量と料金の集計**: トークン数を数えて料金（ドル・円）を出す。`print_usage_summary` / `estimate_cost` など。料金表は `llm_kit/config.py`（2026-11-19 に各社の公式の値と突き合わせた。Vercel AI Gateway の判定モデル `typesafe-ai/jev` などは provider `"gateway"` で引く）。料金表に無いモデルは 0 ドルとして数え、モデルごとに 1 度だけ警告を出す
- **埋め込み**: `embed_query` / `embed_documents`（同期・非同期）
- **リランク**: `rerank`（Bedrock 経由の Cohere か、Cohere の API）

## 入れ方

このリポジトリの直下が、パッケージ `llm_kit` そのもの。clone するときはフォルダ名を `llm_kit` にすると、その親から `import llm_kit` できる。
使うプロバイダの分だけ追加の依存を付ける（`pyproject.toml` の `optional-dependencies`）。

```bash
# GitHub から
uv add "llm-kit[openai,google] @ git+https://github.com/au6000/llm_kit.git"

# 手元の clone を、直しながら使う（llm_kit/ の親で）
pip install -e "./llm_kit[openai,google]"
```

| 追加の依存 | 使う機能 |
|---|---|
| `openai` | `get_llm("openai")`・OpenAI の埋め込み |
| `anthropic` | `get_llm("anthropic")` |
| `google` | `get_llm("google")`・`get_llm("vertex")`・Gemini API の埋め込み |
| `vertex-embeddings` | Vertex の埋め込み |
| `bedrock` | `get_llm("bedrock")`・リランクの Bedrock 経路 |
| `cohere` | リランクの Cohere 経路 |
| `all` | 上の全部 |

`import llm_kit` だけでは LangChain を読まない。`get_llm` などの名前を初めて使った時点で、その部品を読む（`__init__.py`）。
依存を入れていない環境でも `import llm_kit` は通るので、別のプロジェクトの中に置いても、そのプロジェクトの検査の探索で落ちない。

## 使い方

```python
from llm_kit import get_llm, invoke_structured, print_usage_summary
from pydantic import BaseModel

# provider と model を省くと llm_kit/config.py の既定（vertex）を使う
llm = get_llm("vertex", "gemini-3-flash-preview")
print(llm.invoke("こんにちは").text)

class Answer(BaseModel):
    answer: str
    confidence: float

result = invoke_structured("2+2 は？", Answer, provider="openai", model="gpt-6.2-sol")
print(result.answer)

print_usage_summary()
```

## 環境変数

**`get_llm` などを初めて使った時点で、カレントディレクトリの `.env` を読み込む**（`llm_kit/utils.py` の `load_dotenv()`）。

| プロバイダ | 変数 |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google（Gemini API） | `GOOGLE_APPLICATION_CREDENTIALS`（無ければ ADC）・`VERTEX_PROJECT_ID` か `GOOGLE_CLOUD_PROJECT`。API キー（`GOOGLE_API_KEY`）では認証しない |
| Vertex | Google と同じ。加えて `VERTEX_LOCATION` か `GOOGLE_CLOUD_LOCATION`（既定 `global`） |
| Bedrock | `AWS_REGION`（既定 `ap-northeast-1`）と、AWS の通常の認証情報か `AWS_BEARER_TOKEN_BEDROCK` |
| Cohere | `COHERE_API_KEY` |
| Claude Code | `claude` コマンドにパスが通っていること |

## テスト

検査は `tests/` にある。ネットワークも鍵も使わない（モデルは偽物に、Claude Code の CLI はサブプロセスの呼び出しに差し替える）。
`llm_kit/` の**親**で流す。`llm_kit/` の中で流すと、`types.py` などが標準ライブラリの同じ名前の部品を覆い隠す。

```bash
pip install -e "./llm_kit[openai,google]"
python -m unittest discover -s llm_kit/tests -t .
```

依存（langchain-core など）が無ければ検査は丸ごと飛ばし、`google` が無ければ Vertex のストリームを開く検査だけを飛ばす。
別のプロジェクトの直下に `llm_kit/` として置いているなら、そのプロジェクトの `python -m unittest` でも一緒に流れる。

## コメントにある数字

時間切れの秒数や送り直しの回数を決めた理由として、コメントに実測の値（秒数・トークン数）を残している。
値を変えるときは、その理由が今も成り立つかを先に確かめる。
