"""Vertex のストリームを開くところ（`_open_stream`）の送り直しを見る。

応答の最初のかたまりが届く前の時間切れ・接続の失敗だけ 1 回送り直し、流し始めたあとの
失敗は送り直さない（利用者に同じ文が 2 回出ないようにするため）。
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from llm_kit.providers import vertex  # noqa: E402


def _stream(chunks, fail_after=None, exc=None):
    """chunks を順に返す。fail_after 個返したあとに exc を投げる。"""

    async def gen():
        for i, chunk in enumerate(chunks):
            if fail_after is not None and i == fail_after:
                raise exc
            yield chunk
        if fail_after is not None and fail_after >= len(chunks):
            raise exc

    return gen()


class Opener:
    """開くたびに、用意した順にストリーム（または例外）を返す。"""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


async def _collect(iterator):
    return [chunk async for chunk in iterator]


class OpenStream(unittest.TestCase):
    def test_最初のかたまりの前の時間切れは1回だけ送り直す(self):
        opener = Opener(
            _stream(["a"], fail_after=0, exc=httpx.ReadTimeout("slow")),
            _stream(["a", "b"]),
        )
        chunks = asyncio.run(_collect_open(opener))
        self.assertEqual(chunks, ["a", "b"])
        self.assertEqual(opener.calls, 2)

    def test_開くときの接続の失敗も送り直す(self):
        opener = Opener(httpx.ConnectError("down"), _stream(["a"]))
        self.assertEqual(asyncio.run(_collect_open(opener)), ["a"])
        self.assertEqual(opener.calls, 2)

    def test_2回続けて時間切れなら諦めて上げる(self):
        opener = Opener(
            _stream([], fail_after=0, exc=httpx.ReadTimeout("slow")),
            _stream([], fail_after=0, exc=httpx.ReadTimeout("slow")),
        )
        with self.assertRaises(httpx.ReadTimeout):
            asyncio.run(_collect_open(opener))
        self.assertEqual(opener.calls, 2)

    def test_流し始めたあとの失敗は送り直さない(self):
        opener = Opener(
            _stream(["a", "b"], fail_after=1, exc=httpx.ReadTimeout("slow")),
            _stream(["x"]),
        )
        got: list = []

        async def run():
            async for chunk in await vertex._open_stream(opener):
                got.append(chunk)

        with self.assertRaises(httpx.ReadTimeout):
            asyncio.run(run())
        self.assertEqual(got, ["a"])
        self.assertEqual(opener.calls, 1)

    def test_送り直してよい種類でない失敗は送り直さない(self):
        opener = Opener(ValueError("bad request"), _stream(["a"]))
        with self.assertRaises(ValueError):
            asyncio.run(_collect_open(opener))
        self.assertEqual(opener.calls, 1)

    def test_空のストリームは空のまま(self):
        opener = Opener(_stream([]))
        self.assertEqual(asyncio.run(_collect_open(opener)), [])


class _FakeLlm:
    """get_llm の代わり。渡された引数と、ストリームを開いた回数を記録する。"""

    def __init__(self, kwargs, opener):
        self.kwargs = kwargs
        models = type("M", (), {"generate_content_stream": staticmethod(lambda **_: opener())})
        self.client = type("C", (), {"aio": type("A", (), {"models": models})})

    def _prepare_request(self, _messages, **_kwargs):
        return {"config": None}


def _run_stream(bounded: bool, opener):
    from llm_kit import llm as llm_module

    seen: dict = {}
    original = llm_module.get_llm

    def fake_get_llm(_provider, _model, **kwargs):
        seen.update(kwargs)
        return _FakeLlm(kwargs, opener)

    llm_module.get_llm = fake_get_llm
    try:
        async def run():
            async for _ in vertex.stream_tool_calls([], [], model="gemini-x", bounded=bounded):
                pass

        asyncio.run(run())
    finally:
        llm_module.get_llm = original
    return seen


class Bounded(unittest.TestCase):
    """1 ターンに時間がかかる前提の呼び出しは bounded=False で呼び、打ち切りも送り直しもしない。"""

    def test_既定は上限を付ける(self):
        opener = Opener(_stream([]))
        seen = _run_stream(True, opener)
        self.assertEqual(seen.get("timeout"), vertex._REQUEST_TIMEOUT_SEC)
        self.assertEqual(seen.get("max_retries"), vertex._REQUEST_ATTEMPTS)

    def test_boundedがFalseなら上限を渡さない(self):
        opener = Opener(_stream([]))
        seen = _run_stream(False, opener)
        self.assertNotIn("timeout", seen)
        self.assertNotIn("max_retries", seen)

    def test_boundedがFalseなら最初のかたまりの前の時間切れでも送り直さない(self):
        opener = Opener(
            _stream([], fail_after=0, exc=httpx.ReadTimeout("slow")),
            _stream([]),
        )
        with self.assertRaises(httpx.ReadTimeout):
            _run_stream(False, opener)
        self.assertEqual(opener.calls, 1)


async def _collect_open(opener):
    return await _collect(await vertex._open_stream(opener))


if __name__ == "__main__":
    unittest.main()
