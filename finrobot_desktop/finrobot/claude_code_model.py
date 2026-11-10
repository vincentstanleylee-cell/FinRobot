"""Claude subscription model: run FinRobot's LLM steps through the local ``claude`` CLI.

Why this exists
---------------
The Anthropic built-in provider needs an **API key** (pay-per-use billing). A
Claude Pro/Max *subscription* is a different thing: it is used by logging in to
the official Claude Code command-line tool. This module lets FinRobot use that
login **without ever touching the credential** — it only launches the user's own
``claude`` binary in headless mode (``claude -p``) and reads its answer, exactly
as a shell script would. The CLI authenticates itself.

Consequences worth knowing:

- Usage counts against the subscription's rate limits (5-hour / weekly), not an
  API bill. A 13-chapter report makes many LLM calls; heavy use can exhaust the
  limit, and the CLI then returns an error we surface verbatim.
- ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` are scrubbed from the child's
  environment, so the CLI can never silently fall back to API billing.
- The CLI is started with every built-in tool, MCP server, slash command and
  settings file disabled, in an empty temp directory: it acts as a plain text
  model, never as a coding agent that could touch the user's files.

How tools work
--------------
``claude -p`` cannot call *FinRobot's* Python tools or return a structured
``output_type`` natively. So this is a pydantic-ai ``FunctionModel`` that
emulates both with a small JSON protocol in the prompt: when the agent has
tools (function tools or an output tool), the model is told to answer with one
JSON object, either ``{"tool_calls": [{"name", "arguments"}]}`` or — when plain
text is allowed — ordinary prose. When the agent has no tools (most pipeline
steps) the model's reply is passed through verbatim.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import RequestUsage

PROVIDER_ID = "claude-code"

# Fixed, short system prompt (the real instructions travel on stdin, so this
# stays well under the Windows command-line length limit and is a stable,
# cache-friendly prefix across calls).
_CLI_SYSTEM_PROMPT = (
    "You are the language model inside a financial-research agent runtime. "
    "Follow the instructions in the user message exactly. Do not use any tools; "
    "you have none. Reply only with what the instructions ask for."
)

# Env vars that would make the CLI bill an API account instead of the subscription.
_SCRUBBED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")

_DEFAULT_TIMEOUT_S = float(os.environ.get("FINROBOT_CLAUDE_CODE_TIMEOUT", "900"))
_MAX_CONCURRENCY = max(1, int(os.environ.get("FINROBOT_CLAUDE_CODE_CONCURRENCY", "3")))
# Thread-level gate (not an asyncio.Semaphore): the CLI call runs in a worker
# thread, and a threading primitive has no event-loop affinity to get wrong.
_gate = threading.BoundedSemaphore(_MAX_CONCURRENCY)


class ClaudeCodeError(RuntimeError):
    """The ``claude`` CLI could not produce an answer.

    The message deliberately contains the words the Settings probe classifier
    keys on (``authentication`` / ``not found``), so the UI shows the right
    localized reason.
    """


def find_claude_cli() -> str | None:
    """Path to the user's ``claude`` binary, or None when it isn't installed."""
    override = os.environ.get("FINROBOT_CLAUDE_CLI")
    if override and Path(override).exists():
        return override
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    for candidate in (
        home / ".local" / "bin" / "claude.exe",
        home / ".local" / "bin" / "claude",
        home / ".claude" / "local" / "claude",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def claude_code_available() -> bool:
    return find_claude_cli() is not None


# ── CLI invocation ────────────────────────────────────────────────────────────


def _run_cli(prompt: str, model_id: str, timeout: float = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """One blocking ``claude -p`` call; returns the CLI's JSON result object."""
    cli = find_claude_cli()
    if cli is None:
        raise ClaudeCodeError(
            "Claude Code CLI not found. Install Claude Code and sign in with your "
            "Claude subscription (run `claude` once in a terminal)."
        )
    cmd = [
        cli,
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",  # no built-in tools: a plain text model
        "--strict-mcp-config",  # no MCP servers
        "--disable-slash-commands",  # no skills / commands
        "--setting-sources",
        "",  # ignore user/project/local settings, hooks and plugins
        "--no-session-persistence",
        "--system-prompt",
        _CLI_SYSTEM_PROMPT,
    ]
    if model_id:
        cmd += ["--model", model_id]

    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"  # don't inject the user's CLAUDE.md files

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    with _gate, tempfile.TemporaryDirectory(prefix="finrobot-claude-") as cwd:
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no shell, prompt via stdin
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=env,
                timeout=timeout,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError(f"Claude Code CLI timed out after {int(timeout)}s.") from exc

    out = (proc.stdout or "").strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        detail = (proc.stderr or out or f"exit code {proc.returncode}").strip()[:400]
        raise ClaudeCodeError(f"Claude Code CLI returned no usable result: {detail}")

    if data.get("is_error") or proc.returncode != 0:
        msg = str(data.get("result") or proc.stderr or "unknown error").strip()
        low = msg.lower()
        if any(s in low for s in ("not logged in", "log in", "login", "/login", "oauth")):
            raise ClaudeCodeError(
                "Claude Code authentication failed — sign in with your Claude subscription "
                f"(run `claude` once in a terminal). CLI said: {msg[:200]}"
            )
        raise ClaudeCodeError(f"Claude Code CLI error: {msg[:400]}")
    return data


# ── Prompt assembly ───────────────────────────────────────────────────────────


def _user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        parts.append(item if isinstance(item, str) else "[non-text attachment omitted]")
    return "\n".join(parts)


def _tool_block(info: AgentInfo) -> str:
    tools = [*info.function_tools, *info.output_tools]
    lines: list[str] = [
        "## Tool-use protocol",
        "You cannot run tools yourself. To use a tool, reply with ONLY one JSON object — "
        "no prose, no markdown code fences — of this exact shape:",
        '{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments as JSON>}}]}',
        "You may request several independent tools in one reply. The results are returned "
        "to you in the next message.",
    ]
    out_names = [t.name for t in info.output_tools]
    if out_names and not info.allow_text_output:
        lines.append(
            "You MUST deliver your final answer by calling "
            + (f"the tool `{out_names[0]}`" if len(out_names) == 1 else f"one of {out_names}")
            + " with the answer as its arguments. Never reply with plain text."
        )
    elif out_names:
        lines.append(
            f"To finish you may either call {out_names} or simply reply in plain text "
            "(any reply that is not a tool_calls JSON object is treated as your final answer)."
        )
    else:
        lines.append(
            "When you have everything you need, reply in plain text (any reply that is not a "
            "tool_calls JSON object is treated as your final answer)."
        )
    lines.append("\nAvailable tools:")
    for t in tools:
        lines.append(f"\n### {t.name}")
        if t.description:
            lines.append(t.description.strip())
        lines.append("Arguments JSON Schema: " + json.dumps(t.parameters_json_schema))
    return "\n".join(lines)


def build_prompt(messages: list[ModelMessage], info: AgentInfo) -> tuple[str, bool]:
    """Flatten an agent conversation into one prompt. Returns (prompt, tool_mode)."""
    system: list[str] = []
    if info.instructions:
        system.append(info.instructions.strip())
    convo: list[str] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    text = part.content.strip()
                    if text and text not in system:
                        system.append(text)
                elif isinstance(part, UserPromptPart):
                    convo.append("[user]\n" + _user_text(part.content))
                elif isinstance(part, ToolReturnPart):
                    convo.append(f"[tool result: {part.tool_name}]\n{part.model_response_str()}")
                elif isinstance(part, RetryPromptPart):
                    convo.append(f"[retry request]\n{part.model_response()}")
        elif isinstance(msg, ModelResponse):
            for rpart in msg.parts:
                if isinstance(rpart, TextPart):
                    convo.append("[assistant]\n" + rpart.content)
                elif isinstance(rpart, ToolCallPart):
                    convo.append(
                        f"[assistant tool call: {rpart.tool_name}]\n{rpart.args_as_json_str()}"
                    )

    tool_mode = bool(info.function_tools or info.output_tools)
    sections: list[str] = []
    if system:
        sections.append("# Instructions\n" + "\n\n".join(system))
    if tool_mode:
        sections.append(_tool_block(info))
    sections.append("# Conversation\n" + ("\n\n".join(convo) if convo else "(empty)"))
    if tool_mode:
        sections.append("# Your reply\nRespond now, following the tool-use protocol above.")
    else:
        sections.append("# Your reply\nWrite the assistant's next reply now.")
    return "\n\n".join(sections), tool_mode


# ── Reply parsing ─────────────────────────────────────────────────────────────


def _extract_json_object(text: str) -> Any | None:
    """First JSON object in ``text`` (tolerates code fences / leading prose)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def parse_reply(text: str, info: AgentInfo, tool_mode: bool) -> list[TextPart | ToolCallPart]:
    if not tool_mode:
        return [TextPart(content=text)]

    obj = _extract_json_object(text)
    calls: list[ToolCallPart] = []
    if isinstance(obj, dict):
        raw_calls = obj.get("tool_calls")
        if isinstance(raw_calls, list):
            for c in raw_calls:
                if isinstance(c, dict) and isinstance(c.get("name"), str):
                    args = c.get("arguments", c.get("args", {}))
                    calls.append(
                        ToolCallPart(
                            tool_name=c["name"],
                            args=args if isinstance(args, dict) else {},
                            tool_call_id=f"cc_{uuid.uuid4().hex[:12]}",
                        )
                    )
        elif len(info.output_tools) == 1 and not info.allow_text_output:
            # Lenient: the model gave the bare arguments of the single output tool.
            calls.append(
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=obj,
                    tool_call_id=f"cc_{uuid.uuid4().hex[:12]}",
                )
            )
    if calls:  # an unknown tool name is fine: pydantic-ai's retry machinery corrects it
        return list(calls)
    return [TextPart(content=text)]  # final plain-text answer (or a retry trigger)


# ── Model factory ─────────────────────────────────────────────────────────────


def _usage(data: dict[str, Any]) -> RequestUsage:
    try:
        mu = next(iter((data.get("modelUsage") or {}).values()))
        return RequestUsage(
            input_tokens=int(mu.get("inputTokens", 0))
            + int(mu.get("cacheReadInputTokens", 0))
            + int(mu.get("cacheCreationInputTokens", 0)),
            output_tokens=int(mu.get("outputTokens", 0)),
        )
    except (StopIteration, TypeError, ValueError):
        return RequestUsage()


def build_claude_code_model(model_id: str) -> Model:
    """A pydantic-ai Model backed by the local ``claude`` CLI (subscription login)."""
    name = f"{PROVIDER_ID}:{model_id}"

    async def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt, tool_mode = build_prompt(messages, info)
        last: ClaudeCodeError | None = None
        data: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                data = await asyncio.to_thread(_run_cli, prompt, model_id)
                if str(data.get("result", "")).strip():
                    break
                last = ClaudeCodeError("Claude Code CLI returned an empty reply.")
            except ClaudeCodeError as exc:
                last = exc
                msg = str(exc).lower()
                # Retrying cannot fix auth / missing CLI / quota problems.
                if any(s in msg for s in ("authentication", "not found", "limit")) or attempt:
                    raise
            await asyncio.sleep(2)
        if data is None or not str(data.get("result", "")).strip():
            raise last or ClaudeCodeError("Claude Code CLI returned no reply.")
        parts = parse_reply(str(data["result"]), info, tool_mode)
        return ModelResponse(parts=parts, model_name=name, usage=_usage(data))

    async def _stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        response = await _respond(messages, info)
        idx = 0
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    idx: DeltaToolCall(
                        name=part.tool_name,
                        json_args=part.args_as_json_str(),
                        tool_call_id=part.tool_call_id,
                    )
                }
                idx += 1

    return FunctionModel(_respond, stream_function=_stream, model_name=name)
