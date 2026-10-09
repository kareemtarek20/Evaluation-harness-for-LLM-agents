"""Stage 4 tests: the shipped task suites must be valid, and invalid files must fail."""

import json
from pathlib import Path

import pytest

from agenteval.models import MatchType
from agenteval.taskset import (
    classify_directory,
    discover_task_files,
    load_task_file,
    load_tasks,
    validate_task_files,
)

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def suite_files() -> list[Path]:
    return discover_task_files(TASKS_DIR)


def test_task_files_are_discovered() -> None:
    names = [path.name for path in suite_files()]
    assert "basic.json" in names
    assert "extended.json" in names


def test_every_shipped_task_file_is_valid() -> None:
    assert validate_task_files(suite_files()) == []


def test_the_judge_sheet_is_not_mistaken_for_a_task_suite() -> None:
    """tasks/judge_labels.json lives beside the suites but must not load as one."""
    suites, others = classify_directory(TASKS_DIR)
    names = [path.name for path in suites]
    assert "basic.json" in names and "extended.json" in names
    assert [path.name for path in others] == ["judge_labels.json"]
    # ...and the strict loader really would have refused it
    assert validate_task_files([TASKS_DIR / "judge_labels.json"]) != []


@pytest.mark.parametrize("path", sorted(discover_task_files(TASKS_DIR)), ids=lambda p: p.name)
def test_each_file_loads_with_required_fields(path: Path) -> None:
    tasks = load_task_file(path)
    assert tasks
    for task in tasks:
        assert task.id.strip()
        assert task.prompt.strip()
        assert task.expected.strip()
        assert task.category.strip()
        assert isinstance(task.match, MatchType)
        assert task.max_steps >= 1
        if task.match is MatchType.JUDGE:
            assert task.rubric.strip()


def test_basic_suite_covers_the_planned_categories() -> None:
    tasks = load_task_file(TASKS_DIR / "basic.json")
    assert len(tasks) == 10
    categories = {task.category for task in tasks}
    assert categories == {"math", "retrieval", "multi_step", "efficiency", "hallucination"}
    assert len({task.id for task in tasks}) == 10


def test_extended_suite_is_large_and_covers_hard_cases() -> None:
    tasks = load_task_file(TASKS_DIR / "extended.json")
    assert len(tasks) >= 30
    categories = {task.category for task in tasks}
    assert {"ambiguous", "tool_error", "injection", "hallucination"} <= categories
    assert any(task.match is MatchType.JUDGE for task in tasks)
    assert any(task.match is MatchType.CONTAINS_ANY for task in tasks)
    assert any(task.forbidden_tools for task in tasks)


def test_ids_are_unique_across_all_task_files() -> None:
    tasks = load_tasks(suite_files())
    ids = [task.id for task in tasks]
    assert len(ids) == len(set(ids)), [task_id for task_id in ids if ids.count(task_id) > 1]


def test_match_types_are_all_used() -> None:
    tasks = load_tasks(suite_files())
    used = {task.match for task in tasks}
    assert used == {
        MatchType.EXACT,
        MatchType.CONTAINS,
        MatchType.CONTAINS_ANY,
        MatchType.NUMERIC,
        MatchType.JUDGE,
    }


def test_only_known_tools_are_referenced() -> None:
    from agenteval.tools import ToolBox

    known = set(ToolBox().names)
    for task in load_tasks(suite_files()):
        assert set(task.expected_tools) <= known
        assert set(task.forbidden_tools) <= known


def test_write(tmp_path: Path) -> None:
    """Helper coverage below: bad files must raise with a precise message."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing prompt"):
        load_task_file(bad)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    payload = [
        {"id": "a", "prompt": "p", "expected": "1", "match": "numeric"},
        {"id": "a", "prompt": "p", "expected": "2", "match": "numeric"},
    ]
    path = tmp_path / "dupes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_task_file(path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps(payload[:1]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks([path, other])


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    payload = [
        {"id": "a", "prompt": "p", "expected": "1", "match": "numeric", "expected_tools": ["email"]}
    ]
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown tool"):
        load_task_file(path)


def test_judge_without_rubric_rejected(tmp_path: Path) -> None:
    payload = [{"id": "a", "prompt": "p", "expected": "any", "match": "judge"}]
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a rubric"):
        load_task_file(path)


def test_bad_shapes_rejected(tmp_path: Path) -> None:
    not_list = tmp_path / "shape.json"
    not_list.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array"):
        load_task_file(not_list)
    scalars = tmp_path / "scalar.json"
    scalars.write_text(json.dumps(["just a string"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        load_task_file(scalars)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_task_file(broken)
    zero_steps = tmp_path / "steps.json"
    zero_steps.write_text(
        json.dumps(
            [{"id": "a", "prompt": "p", "expected": "1", "match": "numeric", "max_steps": 0}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_steps"):
        load_task_file(zero_steps)


def test_validate_task_files_collects_problems(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps([{"id": "a", "prompt": "p", "expected": "1", "match": "numeric"}]),
        encoding="utf-8",
    )
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    problems = validate_task_files([good, bad])
    assert len(problems) == 1
    assert "bad.json" in problems[0]
