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
                "name": "add_numbers",
                "description": "Add two numbers",
                "params": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
            },
        ),
        (
            concat_strings,
            {
                "name": "concat_strings",
                "description": "Concatenate two strings",
                "params": {
                    "a": {"type": "string", "description": "First string"},
                    "b": {"type": "string", "description": "Second string"},
                },
            },
        ),
    ],
)
def test_generate_function_schema_parameterized(test_function, expected):
    """Test that function schema generation works correctly with different functions."""
    schema = generate_function_schema(test_function)

    # Check the basic structure
    assert schema["type"] == "function"
    assert "function" in schema

    # Check the function details
    function = schema["function"]
    assert function["name"] == expected["name"]
    assert function["description"] == expected["description"]
    assert function["strict"] is True

    # Check parameters
    params = function["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert params["required"] == ["a", "b"]
    assert params["additionalProperties"] is False

    # Check parameter properties
    properties = params["properties"]
    for param_name, param_details in expected["params"].items():
        assert param_name in properties
        assert properties[param_name]["type"] == param_details["type"]
        assert properties[param_name]["description"] == param_details["description"]
