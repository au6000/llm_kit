"""llm_kit のうち、ネットワークも鍵も使わずに確かめられる所を見る。

モデルは手元の偽物に差し替え、Claude Code の CLI はサブプロセスの呼び出しを差し替える。
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import unittest
from typing import Optional
from unittest import mock

try:
    import langchain_core  # noqa: F401
    import pydantic  # noqa: F401
except ImportError:
    raise unittest.SkipTest("llm_kit の依存 (langchain-core など) がない")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from llm_kit import llm, usage_tracker, utils
from llm_kit.exceptions import LLMAPIError, LLMConfigError, LLMResponseError
from llm_kit.providers import claude_code


class _Scripted(BaseChatModel):
    """決めた返事を順に返し、受け取ったメッセージ列を覚える偽のモデル。"""

    replies: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    delay: float = 0.0
    active: int = 0
    peak: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._next(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return self._next(messages)
        finally:
            self.active -= 1


def _use(model):
    """llm.get_llm を、引数によらず model を返す関数に差し替える。"""
    return mock.patch.object(llm, "get_llm", lambda *a, **k: model)


def _blocks(*texts):
    return AIMessage(content=[{"type": "text", "text": t} for t in texts])


class Person(BaseModel):
    user_name: str
    tags: list[str] = []
    note: Optional[str] = None


class PipePerson(BaseModel):
    user_name: str
    note: str | None = None


# ── effort ───────────────────────────────────────────────────────────────


class Effort(unittest.TestCase):
    def test_持たない値は直下へ丸める(self):
        self.assertEqual(llm._clamp_effort("minimal", "openai"), "none")
        self.assertEqual(llm._clamp_effort("max", "vertex"), "high")

    def test_最小値より下しか無ければ最小値へ上げる(self):
        self.assertEqual(llm._clamp_effort("none", "anthropic"), "low")

    def test_表に無いproviderはそのまま(self):
        self.assertEqual(llm._clamp_effort("max", "bedrock"), "max")

    def test_名前は空白と大文字を吸収し未知は捨てる(self):
        self.assertEqual(llm._normalize_effort(" HIGH "), "high")
        with self.assertLogs("llm_kit.llm", level="WARNING"):
            self.assertIsNone(llm._normalize_effort("turbo"))
        self.assertIsNone(llm._normalize_effort(None))


# ── 緩い読み直し ──────────────────────────────────────────────────────────


class LenientParse(unittest.TestCase):
    def test_コードフェンスを外す(self):
        p = utils.try_lenient_parse('```json\n{"user_name": "a"}\n```', Person)
        self.assertEqual(p.user_name, "a")

    def test_前後の地の文を捨てる(self):
        p = utils.try_lenient_parse('はい。{"user_name": "a"} 以上です。{"x": 1}', Person)
        self.assertEqual(p.user_name, "a")

    def test_途中で切れた文字列と括弧を補う(self):
        p = utils.try_lenient_parse('{"user_name": "a", "note": "abc', Person)
        self.assertEqual((p.user_name, p.note), ("a", "abc"))

    def test_途中で切れた配列も補う(self):
        p = utils.try_lenient_parse('{"user_name": "a", "tags": ["x", "y"', Person)
        self.assertEqual(p.tags, ["x", "y"])

    def test_camelCaseの名前を直す(self):
        p = utils.try_lenient_parse('{"userName": "a", "UserName": "b"}', Person)
        self.assertIn(p.user_name, {"a", "b"})

    def test_文字列をリストに割る(self):
        p = utils.try_lenient_parse('{"user_name": "a", "tags": "x、y,\\nz"}', Person)
        self.assertEqual(p.tags, ["x", "y", "z"])

    def test_Optionalの空文字はNone(self):
        p = utils.try_lenient_parse('{"user_name": "a", "note": "  "}', Person)
        self.assertIsNone(p.note)

    def test_縦棒で書いたOptionalの空文字もNone(self):
        p = utils.try_lenient_parse('{"user_name": "a", "note": ""}', PipePerson)
        self.assertIsNone(p.note)

    def test_読めなければNone(self):
        for raw in ["", "   ", "JSON なし", "[1, 2]", '{"note": "x"}']:
            self.assertIsNone(utils.try_lenient_parse(raw, Person), raw)

    def test_送り直しの文は前回の出力を1000字で切ってスキーマを添える(self):
        original = [HumanMessage(content="q")]
        msgs = utils.build_retry_messages(original, "x" * 5000, "壊れている", Person)
        self.assertEqual(msgs[:1], original)
        self.assertEqual(len(msgs), 2)
        self.assertIn("x" * 1000, msgs[1].content)
        self.assertNotIn("x" * 1001, msgs[1].content)
        self.assertIn("user_name", msgs[1].content)


# ── 呼び出し ───────────────────────────────────────────────────────────────


class Invoke(unittest.TestCase):
    def test_systemは先頭に入りhumanが続く(self):
        model = _Scripted(replies=[AIMessage(content="ok")])
        with _use(model):
            self.assertEqual(llm.invoke("q", system_prompt="s"), "ok")
        self.assertEqual(
            [(type(m), m.content) for m in model.seen[0]],
            [(SystemMessage, "s"), (HumanMessage, "q")],
        )

    def test_systemが無ければhumanだけ(self):
        model = _Scripted(replies=[AIMessage(content="ok")])
        with _use(model):
            llm.invoke("q")
        self.assertEqual([type(m) for m in model.seen[0]], [HumanMessage])

    def test_ブロックで返っても文字列にする(self):
        model = _Scripted(replies=[_blocks("a", "b"), _blocks("c")])
        with _use(model):
            self.assertEqual(llm.invoke("q"), "ab")
            self.assertEqual(asyncio.run(llm.ainvoke("q")), "c")

    def test_流しても文字列を返す(self):
        model = _Scripted(replies=[_blocks("a", "b")])
        with _use(model):
            self.assertEqual(list(llm.stream("q")), ["ab"])

    def test_ainvokeもsystemを先頭に入れる(self):
        model = _Scripted(replies=[AIMessage(content="ok")])
        with _use(model):
            self.assertEqual(asyncio.run(llm.ainvoke("q", system_prompt="s")), "ok")
        self.assertEqual([type(m) for m in model.seen[0]], [SystemMessage, HumanMessage])


class StructuredWithRetry(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("asyncio.sleep", new=mock.AsyncMock())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_初回の文にスキーマの指示を付ける(self):
        model = _Scripted(replies=[AIMessage(content='{"user_name": "a"}')])
        with _use(model):
            p = asyncio.run(llm.ainvoke_structured_with_retry("q", Person, system_prompt="s"))
        self.assertEqual(p.user_name, "a")
        first = model.seen[0]
        self.assertEqual([type(m) for m in first], [SystemMessage, HumanMessage])
        self.assertTrue(first[1].content.startswith("q"))
        self.assertIn("user_name", first[1].content)

    def test_読めなければ誤りを添えて送り直す(self):
        model = _Scripted(replies=[AIMessage(content="だめ"), _blocks('{"user_name":', ' "a"}')])
        with _use(model):
            p = asyncio.run(llm.ainvoke_structured_with_retry("q", Person))
        self.assertEqual(p.user_name, "a")
        self.assertEqual(len(model.seen), 2)
        self.assertEqual(len(model.seen[1]), len(model.seen[0]) + 1)
        self.assertIn("だめ", model.seen[1][-1].content)

    def test_最後まで読めなければLLMResponseError(self):
        model = _Scripted(replies=[AIMessage(content="だめ")] * 3)
        with _use(model), self.assertRaises(LLMResponseError) as ctx:
            asyncio.run(llm.ainvoke_structured_with_retry("q", Person, max_retries=3))
        self.assertEqual(len(model.seen), 3)
        self.assertNotIn("だめ", str(ctx.exception))


class Concurrency(unittest.TestCase):
    def tearDown(self):
        llm._llm_semaphore = None

    def _peak(self, call):
        async def run():
            await asyncio.gather(*(call() for _ in range(4)))

        llm.configure_llm_concurrency(1)
        asyncio.run(run())

    def test_上限はainvoke_structured_with_retryに効く(self):
        model = _Scripted(replies=[AIMessage(content='{"user_name": "a"}')] * 4, delay=0.01)
        with _use(model):
            self._peak(lambda: llm.ainvoke_structured_with_retry("q", Person))
        self.assertEqual(model.peak, 1)

    def test_上限はainvokeにも効く(self):
        model = _Scripted(replies=[AIMessage(content="ok")] * 4, delay=0.01)
        with _use(model):
            self._peak(lambda: llm.ainvoke("q"))
        self.assertEqual(model.peak, 1)


# ── 利用量と料金 ───────────────────────────────────────────────────────────


class Usage(unittest.TestCase):
    def test_料金表から100万トークンあたりで数える(self):
        t = usage_tracker.UsageTracker()
        t.add("openai", "gpt-5.2", 1_000_000, 1_000_000, elapsed_time=2.0)
        self.assertAlmostEqual(t.total_cost_usd, 1.75 + 14.00)
        self.assertEqual(t.summary()["call_count"], 1)
        self.assertEqual(t.summary()["total_tokens"], 2_000_000)

    def test_vertexはgoogleの料金表を使う(self):
        t = usage_tracker.UsageTracker()
        t.add("vertex", "gemini-3-flash-preview", 1_000_000, 0)
        self.assertAlmostEqual(t.total_cost_usd, 0.50)

    def test_実コストを渡せば料金表を使わない(self):
        t = usage_tracker.UsageTracker()
        t.add("claude_code", "x", 10, 10, cost_usd=0.25)
        self.assertAlmostEqual(t.total_cost_usd, 0.25)

    def test_料金表に無いモデルは0円で数え警告を出す(self):
        t = usage_tracker.UsageTracker()
        with self.assertLogs("llm_kit", level="WARNING") as logs:
            t.add("openai", "no-such-model", 100, 100)
        self.assertEqual(t.total_cost_usd, 0.0)
        self.assertIn("no-such-model", "\n".join(logs.output))

    def test_resetで0に戻る(self):
        t = usage_tracker.UsageTracker()
        t.add("openai", "gpt-5.2", 10, 10)
        t.reset()
        self.assertEqual(t.summary()["call_count"], 0)
        self.assertEqual(t.details(), [])

    def test_見積もりは記録と同じ式(self):
        e = usage_tracker.estimate_cost("openai", "gpt-5.2", 1_000_000, 1_000_000)
        self.assertAlmostEqual(e["cost_usd"], 15.75)
        b = usage_tracker.estimate_batch_cost("openai", "gpt-5.2", 10, 100_000, 100_000)
        self.assertAlmostEqual(b["cost_usd"], 15.75)
        self.assertEqual(b["num_requests"], 10)

    def test_既定のモデルは料金表にある(self):
        from llm_kit.config import DEFAULT_MODELS, PRICING

        for provider, model in DEFAULT_MODELS.items():
            if provider == "claude_code":  # CLI が実コストを返すので料金表を使わない
                continue
            with self.subTest(provider=provider):
                self.assertIn(model, PRICING["google" if provider == "vertex" else provider])

    def test_料金はどれも0以上の数(self):
        from llm_kit.config import PRICING

        for provider, table in PRICING.items():
            for model, price in table.items():
                with self.subTest(provider=provider, model=model):
                    self.assertEqual(set(price), {"input", "output"})
                    self.assertTrue(all(isinstance(v, (int, float)) and v >= 0 for v in price.values()))

    def test_判定モデルは入力だけで数える(self):
        t = usage_tracker.UsageTracker()
        t.add("gateway", "typesafe-ai/jev", 1_000_000, 5)
        self.assertAlmostEqual(t.total_cost_usd, 0.042)

    def test_モジュールの関数は既定の集計器を使う(self):
        usage_tracker.reset_usage_stats()
        self.addCleanup(usage_tracker.reset_usage_stats)
        usage_tracker.add_usage("openai", "gpt-5.2", 10, 20)
        self.assertEqual(usage_tracker.get_usage_stats()["call_count"], 1)
        self.assertEqual(usage_tracker.get_usage_details()[0]["output_tokens"], 20)

    def test_コールバックはtoken_usageを拾う(self):
        from langchain_core.outputs import LLMResult

        usage_tracker.reset_usage_stats()
        self.addCleanup(usage_tracker.reset_usage_stats)
        cb = usage_tracker.create_usage_callback("openai", "gpt-5.2")
        result = LLMResult(generations=[], llm_output={"token_usage": {"prompt_tokens": 3, "completion_tokens": 4}})
        cb.on_llm_end(result, run_id=None)
        stats = usage_tracker.get_usage_stats()
        self.assertEqual((stats["total_input_tokens"], stats["total_output_tokens"]), (3, 4))


# ── Vertex の宛先 ──────────────────────────────────────────────────────────


class VertexConfig(unittest.TestCase):
    def _env(self, **env):
        keys = ["GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT",
                "VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"]
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        return mock.patch.dict(os.environ, {**clean, **env}, clear=True)

    def test_環境変数のprojectと既定のglobal(self):
        with self._env(VERTEX_PROJECT_ID="p"):
            cfg = utils.get_vertex_config()
        self.assertEqual((cfg["project_id"], cfg["location"]), ("p", "global"))

    def test_locationはVERTEXが先でGOOGLE_CLOUDが次(self):
        with self._env(GOOGLE_CLOUD_PROJECT="p", GOOGLE_CLOUD_LOCATION="us"):
            self.assertEqual(utils.get_vertex_config()["location"], "us")
        with self._env(GOOGLE_CLOUD_PROJECT="p", GOOGLE_CLOUD_LOCATION="us", VERTEX_LOCATION="asia"):
            self.assertEqual(utils.get_vertex_config()["location"], "asia")

    def test_鍵のファイルからprojectを読む(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sa.json")
            with open(path, "w") as f:
                json.dump({"project_id": "from-file"}, f)
            with self._env(GOOGLE_APPLICATION_CREDENTIALS=path):
                self.assertEqual(utils.get_vertex_config()["project_id"], "from-file")
            with self._env(GOOGLE_APPLICATION_CREDENTIALS=path, VERTEX_PROJECT_ID="env"):
                self.assertEqual(utils.get_vertex_config()["project_id"], "env")

    def test_鍵のファイルが無ければLLMConfigError(self):
        with self._env(GOOGLE_APPLICATION_CREDENTIALS="no/such/file.json", VERTEX_PROJECT_ID="p"):
            with self.assertRaises(LLMConfigError):
                utils.get_vertex_config()


# ── Claude Code の CLI ─────────────────────────────────────────────────────


def _payload(result="hi", **extra):
    return json.dumps({
        "subtype": "success", "is_error": False, "result": result,
        "usage": {"input_tokens": 5, "cache_read_input_tokens": 2, "output_tokens": 3},
        "total_cost_usd": 0.01, **extra,
    })


class _Proc:
    def __init__(self, stdout, returncode=0, stderr=b""):
        self._out, self.returncode, self._err = stdout.encode(), returncode, stderr

    async def communicate(self, input=None):
        self.input = input
        return self._out, self._err


class ClaudeCode(unittest.TestCase):
    def setUp(self):
        usage_tracker.reset_usage_stats()
        self.addCleanup(usage_tracker.reset_usage_stats)
        patcher = mock.patch.object(claude_code.shutil, "which", return_value="claude")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, stdout, returncode=0, stderr=""):
        done = subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)
        return mock.patch.object(claude_code.subprocess, "run", return_value=done)

    def test_同期で呼ぶと本文と実コストを返す(self):
        model = claude_code.ChatClaudeCode(model="m")
        with self._run(_payload("hi")) as run:
            msg = model.invoke([SystemMessage(content="s"), HumanMessage(content="q")])
        self.assertEqual(msg.content, "hi")
        args = run.call_args.args[0]
        self.assertEqual(run.call_args.kwargs["input"], "q")
        self.assertEqual(args[args.index("--append-system-prompt") + 1], "s")
        stats = usage_tracker.get_usage_stats()
        self.assertEqual((stats["total_input_tokens"], stats["total_output_tokens"]), (7, 3))
        self.assertAlmostEqual(stats["total_cost_usd"], 0.01)

    def test_非同期でも同じ結果(self):
        model = claude_code.ChatClaudeCode(model="m")
        proc = _Proc(_payload("hi"))
        with mock.patch.object(claude_code.asyncio, "create_subprocess_exec", mock.AsyncMock(return_value=proc)):
            msg = asyncio.run(model.ainvoke([HumanMessage(content="q")]))
        self.assertEqual(msg.content, "hi")
        self.assertEqual(proc.input, "q".encode())

    def test_終了コードが0でなければLLMAPIError(self):
        model = claude_code.ChatClaudeCode(model="m")
        with self._run("", returncode=2, stderr="bad"), self.assertRaises(LLMAPIError):
            model.invoke("q")
        proc = _Proc("", returncode=2, stderr=b"bad")
        with mock.patch.object(claude_code.asyncio, "create_subprocess_exec", mock.AsyncMock(return_value=proc)):
            with self.assertRaises(LLMAPIError):
                asyncio.run(model.ainvoke("q"))

    def test_CLIが見つからなければLLMConfigError(self):
        model = claude_code.ChatClaudeCode(model="m")
        with mock.patch.object(claude_code.shutil, "which", return_value=None), self.assertRaises(LLMConfigError):
            model.invoke("q")

    def test_失敗の知らせはモデルの出力を含めない(self):
        model = claude_code.ChatClaudeCode(model="m")
        out = json.dumps({"subtype": "error_max_turns", "is_error": True, "result": "秘密の発言"})
        with self._run(out), self.assertRaises(LLMAPIError) as ctx:
            model.invoke("q")
        self.assertNotIn("秘密", str(ctx.exception))

    def test_構造化出力はstructured_outputを使う(self):
        model = claude_code.ChatClaudeCode(model="m").with_structured_output(Person)
        with self._run(_payload("", structured_output={"user_name": "a"})) as run:
            p = model.invoke("q")
        self.assertEqual(p.user_name, "a")
        args = run.call_args.args[0]
        self.assertIn("user_name", args[args.index("--json-schema") + 1])

    def test_structured_outputが無ければ本文のJSONを読む(self):
        model = claude_code.ChatClaudeCode(model="m").with_structured_output(Person)
        with self._run(_payload('```json\n{"user_name": "a"}\n```')):
            self.assertEqual(model.invoke("q").user_name, "a")

    def test_include_rawなら読めない理由を返す(self):
        model = claude_code.ChatClaudeCode(model="m").with_structured_output(Person, include_raw=True)
        with self._run(_payload("JSON ではない")):
            out = model.invoke("q")
        self.assertIsNone(out["parsed"])
        self.assertIsNotNone(out["parsing_error"])


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
