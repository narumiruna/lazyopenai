from .client import LazyClient
from .types import BaseTool
from .types import ResponseFormatT


def generate(
    user: str,
    system: str | None = None,
    response_format: type[ResponseFormatT] | None = None,
    tools: list[type[BaseTool]] | None = None,
) -> ResponseFormatT | str:
    client = LazyClient(tools=tools)
    if system:
        client.add_system_message(system)
    client.add_user_message(user)

    return client.generate(response_format=response_format)
