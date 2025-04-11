from typing import Annotated

import pytest
from pydantic import Field

from lazyopenai.schema import generate_function_schema


def add_numbers(
    a: Annotated[float, Field(description="First number")],
    b: Annotated[float, Field(description="Second number")],
) -> float:
    """Add two numbers"""
    return a + b


def concat_strings(
    a: Annotated[str, Field(description="First string")],
    b: Annotated[str, Field(description="Second string")],
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
        ),
        (
            concat_strings,
            {
                "type": "function",
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
        ),
    ],
)
def test_generate_function_schema_parameterized(test_function, expected):
    assert generate_function_schema(test_function) == expected
