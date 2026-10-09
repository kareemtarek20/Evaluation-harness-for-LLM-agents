"""Stage 1 tests: kb_search retrieval, no-match handling, and toolbox dispatch."""

from agenteval.tools import ToolBox


def test_kb_search_returns_the_right_fact() -> None:
    box = ToolBox()
    output, is_error = box.call("kb_search", {"query": "meal per diem domestic"})
    assert not is_error
    assert "hr-expense-meals" in output
    assert "75 USD per day" in output


def test_kb_search_finds_numbers_across_records() -> None:
    box = ToolBox()
    output, _ = box.call("kb_search", {"query": "hotel cap per night"})
    assert "180 USD per night" in output
    output, _ = box.call("kb_search", {"query": "API rate limit requests per minute"})
    assert "500 requests per minute" in output


def test_kb_search_respects_limit() -> None:
    box = ToolBox()
    one, _ = box.call("kb_search", {"query": "security", "limit": 1})
    assert len(one.splitlines()) == 1


def test_kb_search_no_match_is_not_an_error() -> None:
    box = ToolBox()
    output, is_error = box.call("kb_search", {"query": "zylofonium qwerty 9932"})
    assert not is_error
    assert "No matching records" in output


def test_kb_search_stopwords_only_query_is_handled() -> None:
    box = ToolBox()
    output, is_error = box.call("kb_search", {"query": "what is the"})
    assert not is_error
    assert "No matching records" in output


def test_kb_search_rejects_bad_arguments() -> None:
    box = ToolBox()
    assert box.call("kb_search", {})[1]
    assert box.call("kb_search", {"query": "  "})[1]
    assert box.call("kb_search", {"query": "x", "limit": 0})[1]
    assert box.call("kb_search", {"query": "x", "limit": "three"})[1]


def test_unknown_tool_returns_error_data() -> None:
    box = ToolBox()
    output, is_error = box.call("send_email", {"to": " someone@example.com "})
    assert is_error
    assert "unknown tool" in output


def test_toolbox_exposes_schemas_and_names() -> None:
    box = ToolBox()
    assert box.names == ["calculator", "kb_search"]
    by_name = {schema["name"]: schema for schema in box.schemas}
    assert set(by_name) == {"calculator", "kb_search"}
    assert by_name["calculator"]["input_schema"]["required"] == ["expression"]
    assert by_name["kb_search"]["input_schema"]["properties"]["query"]["type"] == "string"


def test_buggy_registered_tool_becomes_error_output() -> None:
    def explode(_arguments: dict) -> tuple[str, bool]:
        msg = "boom"
        raise RuntimeError(msg)

    box = ToolBox({"explode": explode})
    output, is_error = box.call("explode", {})
    assert is_error
    assert "RuntimeError" in output
