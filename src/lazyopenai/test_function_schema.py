from typing import Annotated

import pytest

from lazyopenai.function_schema import generate_function_schema


def add_numbers(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers"""
    return a + b


def concat_strings(
    a: Annotated[str, "First string"],
    b: Annotated[str, "Second string"],
) -> str:
    """Concatenate two strings"""
    return a + b


@pytest.mark.parametrize(
    "test_function,expected",
    [
        (
            add_numbers,
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ),
        (
            concat_strings,
            {
                "type": "function",
                "function": {
                    "name": "concat_strings",
                    "description": "Concatenate two strings",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "description": "First string"},
                            "b": {"type": "string", "description": "Second string"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ),
    ],
)
def test_generate_function_schema_parameterized(test_function, expected):
    assert generate_function_schema(test_function) == expected
