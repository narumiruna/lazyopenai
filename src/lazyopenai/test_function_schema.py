from typing import Annotated

from lazyopenai.function_schema import generate_function_schema


def add_numbers(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers"""
    return a + b


def test_generate_function_schema():
    """Test that function schema generation works correctly."""
    schema = generate_function_schema(add_numbers)

    # Check the basic structure
    assert schema["type"] == "function"
    assert "function" in schema

    # Check the function details
    function = schema["function"]
    assert function["name"] == "add_numbers"
    assert function["description"] == "Add two numbers"
    assert function["strict"] is True

    # Check parameters
    params = function["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert params["required"] == ["a", "b"]
    assert params["additionalProperties"] is False

    # Check parameter properties
    properties = params["properties"]
    assert "a" in properties
    assert "b" in properties

    # Check parameter a
    assert properties["a"]["type"] == "number"
    assert properties["a"]["description"] == "First number"

    # Check parameter b
    assert properties["b"]["type"] == "number"
    assert properties["b"]["description"] == "Second number"
