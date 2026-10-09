"""Load and validate task files.

A task file is either a JSON array of tasks or an object with a ``tasks``
array. Loading is strict on purpose: a typo in a tool name or a duplicated id
silently corrupts a comparison run, so it fails loudly here instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenteval.models import MatchType, Task, task_list

__all__ = [
    "classify_directory",
    "discover_task_files",
    "load_task_file",
    "load_tasks",
    "validate_task_files",
]


def _is_task_suite(path: Path) -> bool:
    """True when a JSON file holds task objects rather than another dataset.

    ``tasks/`` also carries the judge label sheet, whose shape is
    ``{"entries": [...]}``. Loading that as a suite would be a confusing crash,
    so directory scans recognise it and leave it to its own tooling. A file that
    cannot be parsed at all is still returned here, so the loader reports it.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return True
    if isinstance(data, list):
        return bool(data) and all(isinstance(item, dict) and "id" in item for item in data)
    if isinstance(data, dict):
        return "tasks" in data
    return False


def classify_directory(directory: Path | str) -> tuple[list[Path], list[Path]]:
    """Split a directory's JSON into ``(task suites, other datasets)``."""
    candidates = sorted(
        path
        for path in Path(directory).glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )
    suites = [path for path in candidates if _is_task_suite(path)]
    others = [path for path in candidates if path not in suites]
    return suites, others


def discover_task_files(directory: Path | str) -> list[Path]:
    """Return sorted task JSON files in a directory, skipping hidden files."""
    return classify_directory(directory)[0]


def _extract_payloads(path: Path) -> list[dict[str, Any]]:
    """Read a task file and return its task objects."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("tasks")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of tasks (or an object with 'tasks')")
    if not data:
        raise ValueError(f"{path}: contains no tasks")
    problems = [
        f"{path}: task #{index} is not an object"
        for index, item in enumerate(data)
        if not isinstance(item, dict)
    ]
    if problems:
        raise ValueError("; ".join(problems))
    return data


def load_task_file(path: Path | str) -> list[Task]:
    """Parse and validate one task file."""
    source = Path(path)
    payloads = _extract_payloads(source)
    tasks = task_list(payloads)
    _check_unique(tasks, source)
    _check_references(tasks, source)
    return tasks


def load_tasks(paths: list[Path | str] | tuple[Path | str, ...] | Path | str) -> list[Task]:
    """Load several files into one suite and reject cross-file id collisions."""
    single = [paths] if isinstance(paths, (str, Path)) else list(paths)
    tasks: list[Task] = []
    for path in single:
        tasks.extend(load_task_file(path))
    _check_unique(tasks, ", ".join(str(p) for p in single))
    if not tasks:
        raise ValueError(f"no tasks found in {', '.join(str(p) for p in single)}")
    return tasks


def _check_unique(tasks: list[Task], origin: Path | str) -> None:
    """Raise when a task id repeats inside a suite."""
    seen: set[str] = set()
    dupes: list[str] = []
    for task in tasks:
        if task.id in seen:
            dupes.append(task.id)
        seen.add(task.id)
    if dupes:
        raise ValueError(f"{origin}: duplicate task id(s): {', '.join(sorted(set(dupes)))}")


def _check_references(
    tasks: list[Task], origin: Path | str, known_tools: set[str] | None = None
) -> None:
    """Raise on unknown tool names or judge tasks without a rubric."""
    if known_tools is None:
        from agenteval.tools import ToolBox

        known_tools = set(ToolBox().names)
    problems: list[str] = []
    for task in tasks:
        for name in tuple(task.expected_tools) + tuple(task.forbidden_tools):
            if name not in known_tools:
                problems.append(f"{task.id}: unknown tool {name!r}")
        if task.match is MatchType.JUDGE and not task.rubric.strip():
            problems.append(f"{task.id}: match=judge requires a rubric")
        if task.max_steps < 1:
            problems.append(f"{task.id}: max_steps must be >= 1")
    if problems:
        raise ValueError(f"{origin}: " + "; ".join(problems))


def validate_task_files(paths: list[Path | str]) -> list[str]:
    """Return a list of human-readable problems (empty means the suite is valid)."""
    problems: list[str] = []
    for path in paths:
        try:
            load_task_file(path)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            problems.append(str(exc))
    return problems
