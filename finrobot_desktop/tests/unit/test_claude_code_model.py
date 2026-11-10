"""Claude-subscription model: prompt protocol + provider wiring (no real CLI calls)."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

from finrobot import claude_code_model as ccm
from finrobot.config import FinRobotSettings


class Verdict(BaseModel):
    stance: str


def _fake_cli(reply: str, seen: list[str]):
    def run(prompt: str, model_id: str, timeout: float = 0) -> dict:
        seen.append(prompt)
        return {"result": reply, "modelUsage": {"m": {"inputTokens": 3, "outputTokens": 2}}}

    return run


def test_provider_is_builtin_and_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccm, "find_claude_cli", lambda: "claude")
    s = FinRobotSettings(model_name="claude-code:sonnet")
    assert s.is_model_configured is True
    s.validate_runtime_config()  # must not demand an API key
    assert s.create_model().model_name == "claude-code:sonnet"


def test_not_configured_when_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccm, "find_claude_cli", lambda: None)
    s = FinRobotSettings(model_name="claude-code:sonnet")
    assert s.is_model_configured is False
    with pytest.raises(ValueError, match="not found"):
        s.validate_runtime_config()


def test_plain_text_agent_passes_reply_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(ccm, "_run_cli", _fake_cli("hello there", seen))
    agent = Agent(ccm.build_claude_code_model("sonnet"), instructions="Be brief.")
    result = asyncio.run(agent.run("hi"))
    assert result.output == "hello there"
    assert "Be brief." in seen[0] and "Tool-use protocol" not in seen[0]


def test_structured_output_via_tool_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    reply = '```json\n{"tool_calls": [{"name": "final_result", "arguments": {"stance": "hold"}}]}\n```'
    monkeypatch.setattr(ccm, "_run_cli", _fake_cli(reply, seen))
    agent = Agent(ccm.build_claude_code_model("sonnet"), output_type=Verdict)
    result = asyncio.run(agent.run("verdict?"))
    assert result.output == Verdict(stance="hold")
    assert "final_result" in seen[0]


def test_function_tool_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            '{"tool_calls": [{"name": "price", "arguments": {"ticker": "MSFT"}}]}',
            "MSFT trades at 10.",
        ]
    )
    seen: list[str] = []

    def run(prompt: str, model_id: str, timeout: float = 0) -> dict:
        seen.append(prompt)
        return {"result": next(replies)}

    monkeypatch.setattr(ccm, "_run_cli", run)
    agent = Agent(ccm.build_claude_code_model("sonnet"))

    @agent.tool_plain
    def price(ticker: str) -> str:
        """Latest price."""
        return f"{ticker}=10"

    result = asyncio.run(agent.run("price of MSFT?"))
    assert result.output == "MSFT trades at 10."
    assert "[tool result: price]" in seen[1] and "MSFT=10" in seen[1]


def test_api_key_env_is_scrubbed() -> None:
    assert {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} <= set(ccm._SCRUBBED_ENV)
