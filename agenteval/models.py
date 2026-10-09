"""Data models passed between agents, the runner, scoring and reporting.

Everything here is a plain data container: no I/O, no LLM calls, no scoring
rules. That keeps these types importable from tests and from the CLI without
any side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "AgentResult",
    "FailureMode",
    "MatchType",
    "Task",
    "ToolCall",
    "TrialResult",
    "REQUIRED_TASK_FIELDS",
]


class MatchType(StrEnum):
    """How an agent answer is compared against ``Task.expected``."""

    EXACT = "exact"
    CONTAINS = "contains"
    CONTAINS_ANY = "contains_any"
    NUMERIC = "numeric"
    JUDGE = "judge"

    @classmethod
    def parse(cls, value: str | MatchType) -> MatchType:
        """Return the match type for a string, raising on unknown values."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except ValueError as exc:  # pragma: no cover - message only
            valid = ", ".join(member.value for member in cls)
            raise ValueError(f"unknown match type {value!r}; expected one of {valid}") from exc


class FailureMode(StrEnum):
    """Single classification label attached to every failed trial."""

    OK = "ok"
    WRONG_ANSWER = "wrong_answer"
    TOOL_ERROR = "tool_error"
    MISSING_TOOL = "missing_tool"
    FORBIDDEN_TOOL = "forbidden_tool"
    MAX_STEPS = "max_steps"
    AGENT_ERROR = "agent_error"

    @classmethod
    def severity_order(cls) -> tuple[FailureMode, ...]:
        """Return failure modes from most to least severe.

        ``classify`` walks this order so a crashed run that also skipped a tool
        is reported as the crash, not the missing tool.
        """
        return (
            cls.AGENT_ERROR,
            cls.MAX_STEPS,
            cls.FORBIDDEN_TOOL,
            cls.MISSING_TOOL,
            cls.TOOL_ERROR,
            cls.WRONG_ANSWER,
        )


REQUIRED_TASK_FIELDS: tuple[str, ...] = ("id", "prompt", "expected", "match")


@dataclass
class Task:
    """One evaluation case."""

    id: str
    prompt: str
    expected: str
    match: MatchType = MatchType.EXACT
    category: str = "general"
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    max_steps: int = 8
    rubric: str = ""
    tolerance: float = 1e-6
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Task:
        """Build a task from a JSON object, rejecting unknown or missing keys."""
        missing = [key for key in REQUIRED_TASK_FIELDS if key not in payload]
        if missing:
            raise ValueError(f"task {payload.get('id', '<no id>')} is missing {', '.join(missing)}")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"task {payload['id']} has unknown field(s): {', '.join(unknown)}")
        data = dict(payload)
        data["match"] = MatchType.parse(data["match"])
        data["expected_tools"] = tuple(data.get("expected_tools", ()) or ())
        data["forbidden_tools"] = tuple(data.get("forbidden_tools", ()) or ())
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict matching the task file format."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected": self.expected,
            "match": self.match.value,
            "category": self.category,
            "expected_tools": list(self.expected_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "max_steps": self.max_steps,
            "rubric": self.rubric,
            "tolerance": self.tolerance,
            "notes": self.notes,
        }


@dataclass
class ToolCall:
    """One tool invocation made by the agent."""

    tool: str
    input: dict[str, object] = field(default_factory=dict)
    output: str = ""
    is_error: bool = False
    step: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], step: int = 0) -> ToolCall:
        """Build a call record from an agent-supplied mapping."""
        return cls(
            tool=str(payload.get("tool", "")),
            input=dict(payload.get("input", {}) or {}),
            output=str(payload.get("output", "")),
            is_error=bool(payload.get("is_error", False)),
            step=int(payload.get("step", step)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict."""
        return {
            "step": self.step,
            "tool": self.tool,
            "input": self.input,
            "output": self.output,
            "is_error": self.is_error,
        }


@dataclass
class AgentResult:
    """What an agent produced for a single task, before scoring."""

    answer: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""

    @property
    def crashed(self) -> bool:
        """True when the agent raised rather than answering."""
        return self.error is not None

    def tools_used(self) -> list[str]:
        """Return tool names in call order (duplicates kept)."""
        return [call.tool for call in self.tool_calls]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict."""
        return {
            "answer": self.answer,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "steps": self.steps,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


@dataclass
class TrialResult:
    """Scored result for one (task, trial) pair."""

    task_id: str
    trial: int
    agent: str = ""
    result: AgentResult = field(default_factory=AgentResult)
    passed: bool = False
    answer_ok: bool = False
    tools_ok: bool = False
    forbidden_used: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    failure_mode: FailureMode = FailureMode.WRONG_ANSWER
    steps: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    judge_verdict: str = ""
    judge_rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict for the run artifact."""
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "agent": self.agent,
            "passed": self.passed,
            "answer_ok": self.answer_ok,
            "tools_ok": self.tools_ok,
            "forbidden_used": list(self.forbidden_used),
            "missing_required": list(self.missing_required),
            "failure_mode": self.failure_mode.value,
            "steps": self.steps,
            "latency_ms": round(self.latency_ms, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "judge_verdict": self.judge_verdict,
            "judge_rationale": self.judge_rationale,
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrialResult:
        """Rebuild a scored trial from a saved run artifact."""
        raw_result = dict(payload.get("result", {}) or {})
        calls = [ToolCall.from_mapping(call) for call in raw_result.pop("tool_calls", [])]
        result = AgentResult(tool_calls=calls, **raw_result)
        return cls(
            task_id=str(payload["task_id"]),
            trial=int(payload.get("trial", 0)),
            agent=str(payload.get("agent", "")),
            result=result,
            passed=bool(payload.get("passed", False)),
            answer_ok=bool(payload.get("answer_ok", False)),
            tools_ok=bool(payload.get("tools_ok", False)),
            forbidden_used=list(payload.get("forbidden_used", [])),
            missing_required=list(payload.get("missing_required", [])),
            failure_mode=FailureMode(str(payload.get("failure_mode", FailureMode.WRONG_ANSWER))),
            steps=int(payload.get("steps", result.steps)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            input_tokens=int(payload.get("input_tokens", result.input_tokens)),
            output_tokens=int(payload.get("output_tokens", result.output_tokens)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            judge_verdict=str(payload.get("judge_verdict", "")),
            judge_rationale=str(payload.get("judge_rationale", "")),
        )


def task_list(payloads: Sequence[Mapping[str, object]]) -> list[Task]:
    """Parse a sequence of task payloads, surfacing the offending index."""
    tasks: list[Task] = []
    for index, payload in enumerate(payloads):
        try:
            tasks.append(Task.from_dict(payload))
        except ValueError as exc:
            raise ValueError(f"task #{index}: {exc}") from exc
    return tasks
