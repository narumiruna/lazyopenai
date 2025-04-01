import inspect
from typing import Annotated
from typing import Any
from typing import get_args
from typing import get_origin
from typing import get_type_hints

# Mapping from Python types to JSON schema types
PYTHON_TO_JSON_TYPES = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
    None: "null",
    # Add more mappings as needed
}


def generate_function_schema(func) -> dict[str, Any]:  # noqa: C901
    """
    Generate a function schema from a function's type annotations and docstring.
    Uses Annotated type hints to extract parameter descriptions.
    """
    # Extract basic function metadata
    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)
    # Get function description from docstring
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n")[0] if doc else func.__name__

    # Process parameters
    properties = {}
    required = []
    for param_name, param in signature.parameters.items():
        # Check if parameter is required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

        # Extract type and description from Annotated
        param_type = "object"  # default type
        param_desc = param_name  # default description

        if param_name in type_hints:
            type_hint = type_hints[param_name]
            if get_origin(type_hint) is Annotated:
                args = get_args(type_hint)
                if args:
                    # First argument is the actual type
                    base_type = args[0]
                    param_type = PYTHON_TO_JSON_TYPES.get(base_type, "object")
                    # Handle special cases like list[int], dict[str, int], etc.
                    origin = get_origin(base_type)
                    if origin is not None:
                        param_type = PYTHON_TO_JSON_TYPES.get(origin, "object")

                    # Second argument can be a description string or a Field object
                    if len(args) > 1:
                        if isinstance(args[1], str):
                            param_desc = args[1]
                        else:
                            # Try to extract description from a Field object
                            field_obj = args[1]
                            if hasattr(field_obj, "description") and field_obj.description is not None:
                                param_desc = field_obj.description
            else:
                # Handle non-Annotated types
                base_type = type_hint
                param_type = PYTHON_TO_JSON_TYPES.get(base_type, "object")
                # Handle special cases like list[int], dict[str, int], etc.
                origin = get_origin(base_type)
                if origin is not None:
                    param_type = PYTHON_TO_JSON_TYPES.get(origin, "object")

        properties[param_name] = {"type": param_type, "description": param_desc}

    # Create the complete schema
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
