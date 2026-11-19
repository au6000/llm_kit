"""Vertex の利用量のログに、考えた分のトークン数と、最初のかたまりまでの秒数が出るかを見る。

応答が遅いとき、「モデルが長く考えていた」のか「Vertex で待たされた」のかを、ログだけで
分けられるようにするため（2026-10-19）。
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_kit import usage_tracker  # noqa: E402
from llm_kit.providers import vertex  # noqa: E402


def _chunk(prompt=None, candidates=None, thoughts=None, with_meta=True):
    meta = (
        SimpleNamespace(prompt_token_count=prompt, candidates_token_count=candidates, thoughts_token_count=thoughts)
        if with_meta
        else None
    )
    return SimpleNamespace(usage_metadata=meta, candidates=[])


class VertexUsage(unittest.TestCase):
    def test_考えた分は出力に足したうえで別にも返す(self):
        self.assertEqual(vertex._vertex_usage(_chunk(100, 30, 20)), (100, 50, 20))

    def test_考えた分が無ければ0(self):
        self.assertEqual(vertex._vertex_usage(_chunk(100, 30, None)), (100, 30, 0))

    def test_利用量の無いかたまりはNone(self):
        self.assertIsNone(vertex._vertex_usage(_chunk(with_meta=False)))
        self.assertIsNone(vertex._vertex_usage(_chunk(0, 0, 0)))


class UsageLogLine(unittest.TestCase):
    def test_考えた分と最初のかたまりまでの秒数がログに出る(self):
        tracker = usage_tracker.UsageTracker()
        with self.assertLogs("usage", level="INFO") as logs:
            tracker.add("vertex", "gemini-3.7-flash", 70000, 36, 49.6, thinking_tokens=12, first_chunk_seconds=48.7)
        self.assertIn("in=70000 out=36 elapsed=49.60 think=12 first=48.70", logs.output[0])
        record = tracker.details()[-1]
        self.assertEqual(record["thinking_tokens"], 12)
        self.assertEqual(record["first_chunk_seconds"], 48.7)

    def test_渡さなければ今までどおりの行(self):
        tracker = usage_tracker.UsageTracker()
        with self.assertLogs("usage", level="INFO") as logs:
            tracker.add("openai", "gpt-6-luna", 991, 61)
        self.assertTrue(logs.output[0].endswith("in=991 out=61 elapsed=0.00"))


class StreamTiming(unittest.TestCase):
    def test_ストリームの終わりに時間と考えた分を渡す(self):
        async def chunks():
            await asyncio.sleep(0.05)  # 最初のかたまりまで待たされる
            yield _chunk()
            yield _chunk(100, 30, 20)

        async def stream(**_):  # SDK と同じく、待つとストリームが返る
            return chunks()

        fake_llm = SimpleNamespace(
            _prepare_request=lambda *a, **k: {"config": None},
            client=SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream))),
        )

        async def run():
            return [kind async for kind, _ in vertex.stream_tool_calls([], [], model="gemini-3.7-flash", bounded=False)]

        with mock.patch("llm_kit.llm.get_llm", return_value=fake_llm), mock.patch.object(vertex, "add_usage") as add:
            kinds = asyncio.run(run())
        self.assertEqual(kinds, ["done"])
        args, kwargs = add.call_args
        self.assertEqual(args[:4], ("vertex", "gemini-3.7-flash", 100, 50))
        self.assertEqual(kwargs["thinking_tokens"], 20)
        self.assertGreaterEqual(kwargs["first_chunk_seconds"], 0.04)
        self.assertGreaterEqual(kwargs["elapsed_time"], kwargs["first_chunk_seconds"])


if __name__ == "__main__":
    unittest.main()
