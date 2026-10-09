"""Stage 3 tests: agent behaviour that the harness depends on."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agenteval.agents import (
    MOCK_SCRIPTS,
    MODEL_ENV_VAR,
    AnthropicAgent,
    MockAgent,
    resolve_model,
)
from agenteval.models import AgentResult, FailureMode, MatchType, Task, ToolCall
from agenteval.scoring import MAX_STEPS_REASON, score_trial
from agenteval.taskset import load_task_file
from agenteval.tools import ToolBox


def make_task(task_id: str, **overrides: object) -> Task:
    payload: dict[str, object] = {
        "id": task_id,
        "prompt": "What is 2 to the 10th power?",
        "expected": "1024",
        "match": MatchType.NUMERIC.value,
        "expected_tools": ["calculator"],
        "max_steps": 4,
        **overrides,
    }
    return Task.from_dict(payload)


def block(kind: str, **fields: Any) -> SimpleNamespace:
    """Mimic an Anthropic content block closely enough for the loop."""
    return SimpleNamespace(type=kind, **fields)


def response(
    content: list[Any], *, input_tokens: int = 100, output_tokens: int = 20
) -> SimpleNamespace:
    """Mimic a messages.create() response."""
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeClient:
    """Returns queued responses; repeats the last one forever (a looping model)."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self._index = 0
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        snapshot = dict(kwargs)
        snapshot["messages"] = [dict(message) for message in kwargs.get("messages", [])]
        self.calls.append(snapshot)
        item = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return item


def tool_use(name: str, arguments: dict[str, Any], call_id: str) -> SimpleNamespace:
    """A tool_use block."""
    return block("tool_use", name=name, input=arguments, id=call_id)


def test_resolve_model_uses_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=MODEL_ENV_VAR):
        resolve_model()
    monkeypatch.setenv(MODEL_ENV_VAR, "claude-sonnet-4-5")
    assert resolve_model() == "claude-sonnet-4-5"
    assert resolve_model("claude-haiku-4-5") == "claude-haiku-4-5"


def test_mock_agent_is_deterministic() -> None:
    task = make_task("kb_meals", expected="75", match="numeric", expected_tools=["kb_search"])
    first = MockAgent("v1").run(task, ToolBox())
    second = MockAgent("v1").run(task, ToolBox())
    assert first == second


def test_mock_v1_power_passes_and_v2_regresses_it() -> None:
    task = make_task("calc_power")
    v1 = MockAgent("v1").run(task, ToolBox())
    v2 = MockAgent("v2").run(task, ToolBox())
    assert score_trial(task, v1)[2] is True
    answer_ok, _tools, passed, mode, *_ = score_trial(task, v2)
    assert (answer_ok, passed) == (False, False)
    assert mode is FailureMode.TOOL_ERROR
    assert v2.tool_calls[0].is_error is True


def test_mock_v2_fixes_retrieval_that_v1_missed() -> None:
    task = make_task(
        "kb_retention",
        expected="90",
        match="numeric",
        expected_tools=["kb_search"],
        prompt="How long are application logs kept?",
    )
    assert not score_trial(task, MockAgent("v1").run(task, ToolBox()))[2]
    assert score_trial(task, MockAgent("v2").run(task, ToolBox()))[2]


def test_mock_unknown_task_falls_back_deterministically() -> None:
    task = make_task("brand_new_id")
    result = MockAgent("v2").run(task, ToolBox())
    assert result.finish_reason == "stop"
    assert not score_trial(task, result)[2]


def test_mock_tokens_scale_with_rounds() -> None:
    task = make_task("calc_power")
    result = MockAgent("v1").run(task, ToolBox())
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_anthropic_agent_records_tool_error_passthrough() -> None:
    client = FakeClient(
        [
            response([tool_use("calculator", {"expression": "2 ^ 10"}, "call_1")]),
            response([block("text", text="The tool rejected that.")]),
        ]
    )
    agent = AnthropicAgent(model="claude-sonnet-4-5", client=client)
    result = agent.run(make_task("calc_power"), ToolBox())
    assert result.tool_calls[0].is_error is True
    assert "unsafe expression" in result.tool_calls[0].output
    assert result.answer == "The tool rejected that."
    assert result.steps == 2
    assert (result.input_tokens, result.output_tokens) == (200, 40)
    assert result.model == "claude-sonnet-4-5"


def test_anthropic_agent_sends_tool_results_back() -> None:
    client = FakeClient(
        [
            response([tool_use("kb_search", {"query": "meal per diem domestic"}, "call_1")]),
            response([block("text", text="75 USD per day")]),
        ]
    )
    agent = AnthropicAgent(model="claude-sonnet-4-5", client=client)
    result = agent.run(make_task("kb_meals"), ToolBox())
    assert len(client.calls) == 2
    follow_up = client.calls[1]["messages"]
    assert len(follow_up) == 3
    assert follow_up[1]["content"][0].type == "tool_use"
    assert follow_up[2]["role"] == "user"
    tool_result = follow_up[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert "hr-expense-meals" in tool_result["content"]
    assert tool_result["is_error"] is False
    assert result.answer == "75 USD per day"


def test_max_steps_is_enforced_by_a_forever_looping_agent() -> None:
    looping = FakeClient([response([tool_use("calculator", {"expression": "1 + 1"}, "c")])])
    agent = AnthropicAgent(model="claude-sonnet-4-5", client=looping)
    task = make_task("loop", max_steps=3)
    result = agent.run(task, ToolBox())
    assert result.steps == 3
    assert len(result.tool_calls) == 3
    assert result.finish_reason == MAX_STEPS_REASON
    assert classify_max(task, result) is FailureMode.MAX_STEPS


class NeverStopsAgent:
    """A stub agent that ignores its budget entirely."""

    name = "stub:never-stops"

    def run(self, task: Task, toolbox: ToolBox) -> AgentResult:
        calls = [
            ToolCall(step=index, tool="calculator", output="1")
            for index in range(task.max_steps * 3)
        ]
        return AgentResult(
            answer="", tool_calls=calls, steps=len(calls), finish_reason=MAX_STEPS_REASON
        )


def classify_max(task: Task, result: Any) -> FailureMode:
    """Helper: classify a result with no answer."""
    return score_trial(task, result)[3]


def test_stub_agent_looping_forever_is_max_steps() -> None:
    task = make_task("loop", max_steps=4)
    result = NeverStopsAgent().run(task, ToolBox())
    assert result.steps > task.max_steps
    assert classify_max(task, result) is FailureMode.MAX_STEPS


def test_anthropic_agent_exception_becomes_agent_error() -> None:
    def explode(**_kwargs: Any) -> SimpleNamespace:
        msg = "401 bad key"
        raise RuntimeError(msg)

    client = SimpleNamespace(messages=SimpleNamespace(create=explode))
    agent = AnthropicAgent(model="claude-sonnet-4-5", client=client)
    result = agent.run(make_task("calc_power"), ToolBox())
    assert result.crashed
    assert "RuntimeError" in (result.error or "")
    assert score_trial(make_task("calc_power"), result)[3] is FailureMode.AGENT_ERROR


def test_mock_scripts_cover_the_basic_suite() -> None:
    """The regression demo only works if every basic task is scripted in both versions."""
    basic = Path(__file__).resolve().parent.parent / "tasks" / "basic.json"
    task_ids = {task.id for task in load_task_file(basic)}
    for version, script in MOCK_SCRIPTS.items():
        missing = task_ids - set(script)
        assert not missing, f"mock:{version} has no script for {sorted(missing)}"


def test_unknown_mock_version_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mock version"):
        MockAgent("v3")


def test_anthropic_agent_uses_advertised_schemas() -> None:
    client = FakeClient([response([block("text", text="done")])])
    agent = AnthropicAgent(model="claude-sonnet-4-5", client=client, system="be nice")
    result = agent.run(make_task("calc_power"), ToolBox())
    assert result.answer == "done"
    sent = client.calls[0]
    assert sent["model"] == "claude-sonnet-4-5"
    assert sent["system"] == "be nice"
    assert {tool["name"] for tool in sent["tools"]} == {"calculator", "kb_search"}
    assert sent["temperature"] == 0.0
    assert sent["messages"][0]["content"] == make_task("calc_power").prompt
