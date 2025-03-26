from collections.abc import Callable

from .chat import Chat
from .chat import ResponseFormatT


def generate(
    messages: str | list[str],
    system: str | None = None,
    response_format: type[ResponseFormatT] | None = None,
    tools: list[Callable] | None = None,
) -> ResponseFormatT | str:
    client = Chat(tools=tools)
    if system:
        client.add_system_message(system)

    if isinstance(messages, str):
        messages = [messages]

    for message in messages:
        client.add_user_message(message)

    return client.create(response_format=response_format)


def create_chat(tools: list[Callable] | None = None) -> Chat:
    return Chat(tools=tools)
