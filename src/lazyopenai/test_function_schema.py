from typing import Annotated

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


def test_generate_function_schema_concat_strings():
    """Test that function schema generation works correctly for concat_strings."""
    schema = generate_function_schema(concat_strings)

    # Check the basic structure
    assert schema["type"] == "function"
    assert "function" in schema

    # Check the function details
    function = schema["function"]
    assert function["name"] == "concat_strings"
    assert function["description"] == "Concatenate two strings"
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
    assert properties["a"]["type"] == "string"
    assert properties["a"]["description"] == "First string"

    # Check parameter b
    assert properties["b"]["type"] == "string"
    assert properties["b"]["description"] == "Second string"
