from .client import LazyClient
from .client import ResponseFormatT
from .types import LazyTool


def generate(
    user: str,
    system: str | None = None,
    response_format: type[ResponseFormatT] | None = None,
    tools: list[type[LazyTool]] | None = None,
) -> ResponseFormatT | str:
    client = LazyClient(tools=tools)
    if system:
        client.add_message(system, role="system")
    client.add_message(user, role="user")

    return client.generate(response_format=response_format)
