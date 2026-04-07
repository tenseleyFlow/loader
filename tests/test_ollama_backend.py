"""Focused tests for Ollama text tool parsing."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx")

from loader.llm.base import StreamChunk
from loader.llm.ollama import OllamaBackend


class FakeResponse:
    """Small response stub for Ollama complete() tests."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")


class FakeClient:
    """Small async client stub for Ollama complete() tests."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)

    async def post(self, url: str, json: dict) -> FakeResponse:  # noqa: A002
        assert self.responses, f"unexpected Ollama POST to {url}"
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


class FakeStreamResponse:
    """Small streaming response stub for _stream_response tests."""

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads

    async def aiter_lines(self):
        for payload in self._payloads:
            yield json.dumps(payload)


@pytest.mark.asyncio
async def test_ollama_complete_uses_shared_parser_with_allowed_tool_names() -> None:
    backend = OllamaBackend()

    async def fake_describe_model() -> None:
        return None

    backend.describe_model = fake_describe_model  # type: ignore[method-assign]
    backend._client = FakeClient(
        [
            FakeResponse(
                {
                    "message": {
                        "content": (
                            '[calls askuserquestion tool with: '
                            'question="Which path should we take?"]'
                        )
                    },
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                }
            )
        ]
    )

    response = await backend.complete(
        messages=[],
        tools=[{"name": "AskUserQuestion"}, {"name": "TodoWrite"}],
    )

    assert response.content == ""
    assert response.tool_calls[0].name == "AskUserQuestion"
    assert response.tool_calls[0].arguments == {
        "question": "Which path should we take?"
    }
    await backend.close()


@pytest.mark.asyncio
async def test_ollama_stream_response_uses_shared_parser_for_text_tool_calls() -> None:
    backend = OllamaBackend()

    chunks = [
        chunk
        async for chunk in backend._stream_response(
            FakeStreamResponse(
                [
                    {
                        "message": {
                            "content": (
                                '[calls askuserquestion tool with: '
                                'question="Which path should we take?"]'
                            )
                        },
                        "done": False,
                    },
                    {
                        "message": {"content": ""},
                        "done": True,
                        "prompt_eval_count": 4,
                        "eval_count": 2,
                    },
                ]
            ),
            tools=[{"name": "AskUserQuestion"}, {"name": "TodoWrite"}],
        )
    ]

    final_chunk = chunks[-1]
    assert isinstance(final_chunk, StreamChunk)
    assert final_chunk.tool_calls[0].name == "AskUserQuestion"
    assert final_chunk.tool_calls[0].arguments == {
        "question": "Which path should we take?"
    }
    await backend.close()
