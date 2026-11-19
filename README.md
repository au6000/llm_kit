# llm-kit

LLM を呼ぶときに毎回要る機能を、ひととおり先に作っておいたライブラリ。
中身は LangChain の上に作ってあり、`get_llm()` が返すのは LangChain のチャットモデルなので、LangGraph にもそのまま渡せる。

## できること

- **プロバイダの切り替え**: `get_llm(provider, model)` の引数を変えるだけで、OpenAI・Anthropic・Google（Gemini API）・Vertex・Bedrock・Claude Code（CLI 経由）を切り替える
- **構造化出力**: `invoke_structured` / `ainvoke_structured`。`ainvoke_structured_with_retry` は、返ってきた JSON が崩れていたらまず緩く読み直し、それでも読めなければ誤りを添えて送り直す（既定で最大 3 回）
- **ストリーミング**: `stream`、およびツール呼び出しの引数を届いた分から受け取る `astream_tool_calls`（OpenAI・Vertex）
- **同時に呼ぶ数の上限**: `configure_llm_concurrency`
- **利用量と料金の集計**: トークン数を数えて料金（ドル・円）を出す。`print_usage_summary` / `estimate_cost` など。料金表は `llm_kit/config.py`
- **埋め込み**: `embed_query` / `embed_documents`（同期・非同期）
- **リランク**: `rerank`（Bedrock 経由の Cohere か、Cohere の API）
- **ベクトル検索の置き場**: `SqliteBackend`（sqlite-vec）

## 入れ方

使うプロバイダの分だけ追加の依存を付ける（`pyproject.toml` の `optional-dependencies`）。

```bash
# GitHub から
uv add "llm-kit[openai,google] @ git+https://github.com/au6000/llm_kit.git"

# 全部入れる
uv add "llm-kit[all] @ git+https://github.com/au6000/llm_kit.git"

# 手元に clone したものを、直しながら使う
uv add --editable "llm-kit[openai,google] @ file:///path/to/llm-kit"
```

| 追加の依存 | 使う機能 |
|---|---|
| `openai` | `get_llm("openai")`・OpenAI の埋め込み |
| `anthropic` | `get_llm("anthropic")` |
| `google` | `get_llm("google")`・`get_llm("vertex")`・Gemini API の埋め込み |
| `vertex-embeddings` | Vertex の埋め込み |
| `bedrock` | `get_llm("bedrock")`・リランクの Bedrock 経路 |
| `cohere` | リランクの Cohere 経路 |
| `sqlite` | `SqliteBackend` |

## 使い方

```python
from llm_kit import get_llm, invoke_structured, print_usage_summary
from pydantic import BaseModel

# provider と model を省くと llm_kit/config.py の既定（vertex）を使う
llm = get_llm("vertex", "gemini-3-flash-preview")
print(llm.invoke("こんにちは").content)

class Answer(BaseModel):
    answer: str
    confidence: float

result = invoke_structured("2+2 は？", Answer, provider="openai", model="gpt-5.2")
print(result.answer)

print_usage_summary()
```

## 環境変数

**import した時点で、カレントディレクトリの `.env` を読み込む**（`llm_kit/utils.py` の `load_dotenv()`）。

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

```bash
uv sync --extra openai --extra google
uv run python -m unittest discover -s tests
```

## コメントにある数字

時間切れの秒数や送り直しの回数を決めた理由として、コメントに実測の値（秒数・トークン数）を残している。
値を変えるときは、その理由が今も成り立つかを先に確かめる。
