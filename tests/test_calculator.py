"""Stage 1 tests: the calculator sandbox must compute correctly and refuse abuse."""

import pytest

from agenteval.tools import ToolBox, calculator_tool, safe_calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "5"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("10 / 4", "2.5"),
        ("10 // 4", "2"),
        ("17 % 5", "2"),
        ("2 ** 10", "1024"),
        ("-3 + 1", "-2"),
        ("3.5 * 2", "7"),
        ("((1+2)*(3+4))", "21"),
        ("1000 - 250 - 75", "675"),
        ("75 + 180 * 3", "615"),
    ],
)
def test_calculator_correct(expression: str, expected: str) -> None:
    assert safe_calculate(expression) == expected


def test_calculator_rejects_import_and_calls() -> None:
    for attack in (
        "__import__('os').system('calc')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "().__class__.__mro__",
        "[i for i in range(3)]",
        "1 if 1 else 2",
        "x + 1",
    ):
        output, is_error = calculator_tool({"expression": attack})
        assert is_error, attack
        assert "error:" in output


def test_caret_is_xor_and_is_rejected() -> None:
    # 2^10 must not silently mean exponentiation; BitXor is not whitelisted.
    output, is_error = calculator_tool({"expression": "2 ^ 10"})
    assert is_error
    assert "unsupported syntax" in output or "unsafe expression" in output


def test_huge_exponents_rejected() -> None:
    for expression in ("10 ** 1000000", "9 ** 9 ** 9", "2 ** 65", "1000001 ** 3"):
        output, is_error = calculator_tool({"expression": expression})
        assert is_error, expression
        assert "too large" in output or "literal" in output


def test_moderate_exponent_allowed() -> None:
    assert safe_calculate("2 ** 64") == str(2**64)


def test_division_by_zero_is_error_not_crash() -> None:
    output, is_error = calculator_tool({"expression": "1 / 0"})
    assert is_error
    assert "division by zero" in output


def test_syntax_errors_and_empty_are_data() -> None:
    for expression in ("", "   ", "2 +", "(1 + 2", "1 1"):
        output, is_error = calculator_tool({"expression": expression})
        assert is_error, expression
        assert output.startswith("error:")


def test_string_and_bool_operands_rejected() -> None:
    for expression in ("'a' + 'b'", "True + 1", "None", "b'ab' * 2"):
        assert calculator_tool({"expression": expression})[1], expression


def test_overlong_expression_rejected() -> None:
    assert calculator_tool({"expression": "1+" * 200})[1]


def test_calculator_via_toolbox_validates_arguments() -> None:
    box = ToolBox()
    assert box.call("calculator", {})[1]
    assert box.call("calculator", {"expression": 5})[1]
    assert box.call("calculator", {"expression": "6", "extra": 1})[1]
    assert box.call("calculator", {"expression": "2 * 21"}) == ("42", False)
