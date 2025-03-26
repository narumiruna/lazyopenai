import inspect
from typing import Annotated
from typing import Any
from typing import get_args
from typing import get_origin
from typing import get_type_hints


def generate_function_schema(func) -> dict[str, Any]:
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
                    param_type = args[0].__name__
                    # Second argument is typically the description
                    if len(args) > 1 and isinstance(args[1], str):
                        param_desc = args[1]
            else:
                # Handle non-Annotated types
                param_type = type_hint.__name__

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
