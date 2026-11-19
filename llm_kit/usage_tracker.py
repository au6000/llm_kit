"""
LLMの使用量・コストを記録・集計するモジュール

記録の入れ物（UsageTracker とグローバル関数群）と、
LangChain の実行イベントからその数字を拾うコールバックを持つ。
"""

import json
from collections import deque
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .config import USD_TO_JPY
from .types import UsageRecord
from .utils import get_pricing

#: 明細（`details` で返すもの）を持つ件数の上限。
_MAX_RECORDS = 1000


class UsageTracker:
    """トークン使用量とコストを記録（スレッドセーフ）"""

    def __init__(self):
        # 明細は直近の分だけ持つ。サーバーのように長く動くプロセスで、呼んだ回数に比例して
        # メモリが増えないようにする。合計と回数は別に持つので、明細を捨てても変わらない。
        self._records: deque[UsageRecord] = deque(maxlen=_MAX_RECORDS)
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._total_elapsed_time: float = 0.0
        self._lock = threading.Lock()

    def add(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        elapsed_time: float = 0.0,
        cost_usd: Optional[float] = None,
        thinking_tokens: Optional[int] = None,
        first_chunk_seconds: Optional[float] = None,
    ):
        """使用量を追加（スレッドセーフ）

        cost_usd を指定した場合は料金表からの推定ではなく、その実コストを記録する
        （Claude Code CLI のように実コストが取得できるプロバイダー向け）。

        thinking_tokens は output_tokens の内数（考えるのに使った分）、first_chunk_seconds は
        応答の最初のかたまりまでの秒数。どちらも分かるプロバイダーだけが渡す。応答が遅かった
        ときに「モデルが長く考えていた」のか「プロバイダーで待たされた」のかを分けるため。
        """
        if cost_usd is not None:
            cost = cost_usd
        else:
            pricing = get_pricing(provider, model)
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        record = UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            elapsed_time=elapsed_time,
            thinking_tokens=thinking_tokens,
            first_chunk_seconds=first_chunk_seconds,
        )

        # 呼び出しごとに 1 行出す。応答が遅かったとき、考えた分（think）と最初のかたまりまでの
        # 秒数（first）から、モデルが長く考えていたのか、プロバイダーで待たされたのかを分ける
        # （1 回の呼び出しが 48 秒待ったのに、out は 36 トークンだった実測がある）。
        import logging as _logging
        extra = ""
        if thinking_tokens is not None:
            extra += f" think={thinking_tokens}"
        if first_chunk_seconds is not None:
            extra += f" first={first_chunk_seconds:.2f}"
        _logging.getLogger("usage").info(
            "[usage] model=%s in=%d out=%d elapsed=%.2f%s",
            model, input_tokens, output_tokens, elapsed_time, extra,
        )

        with self._lock:
            self._records.append(record)
            self._call_count += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost_usd += cost
            self._total_elapsed_time += elapsed_time

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_cost_jpy(self) -> float:
        return self._total_cost_usd * USD_TO_JPY

    @property
    def total_elapsed_time(self) -> float:
        return self._total_elapsed_time

    def summary(self) -> dict:
        """サマリーを返す"""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_cost_jpy": round(self.total_cost_jpy, 2),
            "total_elapsed_time": round(self.total_elapsed_time, 2),
        }

    def details(self) -> list[dict]:
        """詳細な使用量履歴を取得"""
        return [
            {
                "provider": r.provider,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "elapsed_time": r.elapsed_time,
                "timestamp": r.timestamp,
                "thinking_tokens": r.thinking_tokens,
                "first_chunk_seconds": r.first_chunk_seconds,
            }
            for r in self._records
        ]

    def reset(self):
        """統計をリセット"""
        self._records = deque(maxlen=_MAX_RECORDS)
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._total_elapsed_time = 0.0

    def print_summary(self):
        """使用量サマリーを出力"""
        print(f"\n{'='*40}")
        print("LLM Usage Summary")
        print('='*40)
        print(f"API呼び出し回数: {self.call_count}")
        print(f"入力トークン:    {self.total_input_tokens:,}")
        print(f"出力トークン:    {self.total_output_tokens:,}")
        print(f"合計トークン:    {self.total_tokens:,}")
        print(f"合計時間:        {self.total_elapsed_time:.2f}秒")
        print(f"推定コスト:      ${self.total_cost_usd:.4f} (約{self.total_cost_jpy:.0f}円)")
        print('='*40 + "\n")

    def save_to_file(self, filepath: str):
        """使用量をJSONファイルに保存"""
        data = {
            "summary": self.summary(),
            "details": self.details(),
        }
        Path(filepath).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load_from_file(self, filepath: str):
        """JSONファイルから使用量を読み込み（累積）"""
        data = json.loads(Path(filepath).read_text())
        for record in data.get("details", []):
            self.add(
                provider=record["provider"],
                model=record["model"],
                input_tokens=record["input_tokens"],
                output_tokens=record["output_tokens"],
            )


# =============================================================================
# グローバルインスタンス
# =============================================================================
_default_tracker = UsageTracker()


def get_default_tracker() -> UsageTracker:
    """デフォルトトラッカーを取得"""
    return _default_tracker


def add_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    elapsed_time: float = 0.0,
    cost_usd: Optional[float] = None,
    thinking_tokens: Optional[int] = None,
    first_chunk_seconds: Optional[float] = None,
):
    """使用量を追加（デフォルトトラッカー）"""
    _default_tracker.add(
        provider,
        model,
        input_tokens,
        output_tokens,
        elapsed_time,
        cost_usd=cost_usd,
        thinking_tokens=thinking_tokens,
        first_chunk_seconds=first_chunk_seconds,
    )


def get_usage_stats() -> dict:
    """現在の使用量統計を取得"""
    return _default_tracker.summary()


def get_usage_details() -> list[dict]:
    """詳細な使用量履歴を取得"""
    return _default_tracker.details()


def reset_usage_stats():
    """使用量統計をリセット"""
    _default_tracker.reset()


def print_usage_summary():
    """使用量サマリーを出力"""
    _default_tracker.print_summary()


def save_usage(filepath: str):
    """使用量をファイルに保存"""
    _default_tracker.save_to_file(filepath)


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """
    コストを事前見積もり

    Returns:
        dict: {
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "cost_usd": float,
            "cost_jpy": float,
        }
    """
    pricing = get_pricing(provider, model)
    cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    cost_jpy = cost_usd * USD_TO_JPY

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_jpy": round(cost_jpy, 2),
    }


def estimate_batch_cost(
    provider: str,
    model: str,
    num_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
) -> dict:
    """
    バッチ処理のコストを事前見積もり

    Args:
        provider: プロバイダー名
        model: モデル名
        num_requests: リクエスト数
        avg_input_tokens: 1リクエストあたりの平均入力トークン数
        avg_output_tokens: 1リクエストあたりの平均出力トークン数

    Returns:
        dict: 見積もり結果
    """
    total_input = num_requests * avg_input_tokens
    total_output = num_requests * avg_output_tokens

    result = estimate_cost(provider, model, total_input, total_output)
    result["num_requests"] = num_requests
    result["avg_input_tokens"] = avg_input_tokens
    result["avg_output_tokens"] = avg_output_tokens

    return result


def print_estimate(estimate: dict):
    """見積もり結果を出力"""
    print(f"\n{'='*40}")
    print("Cost Estimate (見積もり)")
    print('='*40)
    if "num_requests" in estimate:
        print(f"リクエスト数:    {estimate['num_requests']:,}")
        print(f"平均入力トークン: {estimate['avg_input_tokens']:,}")
        print(f"平均出力トークン: {estimate['avg_output_tokens']:,}")
    print(f"合計入力トークン: {estimate['input_tokens']:,}")
    print(f"合計出力トークン: {estimate['output_tokens']:,}")
    print(f"合計トークン:    {estimate['total_tokens']:,}")
    print(f"推定コスト:      ${estimate['cost_usd']:.4f} (約{estimate['cost_jpy']:.0f}円)")
    print('='*40 + "\n")


# =============================================================================
# LangChain用コールバック（使用量トラッキング）
# =============================================================================
class UsageTrackerCallback(BaseCallbackHandler):
    """
    LLM呼び出しの使用量・時間を記録するコールバック

    並列呼び出し対応:
        - run_idごとに開始時刻を保持
        - スレッドセーフな add_usage に記録
    """

    def __init__(self, provider: str, model: str):
        """
        Args:
            provider: プロバイダー名
            model: モデル名
        """
        self.provider = provider
        self.model = model
        self._start_times: Dict[UUID, float] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """LLM呼び出し開始時"""
        self._start_times[run_id] = time.time()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """LLM呼び出し終了時"""
        elapsed_time = time.time() - self._start_times.pop(run_id, time.time())

        # トークン数を取得
        input_tokens = 0
        output_tokens = 0

        # llm_outputから取得を試みる
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

            # usage キーの場合もある（Anthropic等）
            usage = response.llm_output.get("usage", {})
            if usage:
                input_tokens = input_tokens or usage.get("input_tokens", 0)
                output_tokens = output_tokens or usage.get("output_tokens", 0)

        # generationsからusage_metadataを取得（Vertex AI等）
        if (input_tokens == 0 or output_tokens == 0) and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                        metadata = gen.message.usage_metadata
                        if metadata:
                            input_tokens = input_tokens or metadata.get("input_tokens", 0)
                            output_tokens = output_tokens or metadata.get("output_tokens", 0)

        # トラッカーに記録
        if input_tokens > 0 or output_tokens > 0:
            add_usage(
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                elapsed_time=elapsed_time,
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """エラー時はstart_timesをクリーンアップ"""
        self._start_times.pop(run_id, None)


def create_usage_callback(provider: str, model: str) -> UsageTrackerCallback:
    """使用量トラッキング用コールバックを作成"""
    return UsageTrackerCallback(provider=provider, model=model)
