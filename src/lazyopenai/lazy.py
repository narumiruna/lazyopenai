from collections.abc import Callable

from .agent import Agent
from .agent import ResponseFormatT


def generate(
    messages: str | list[str],
    instruction: str | None = None,
    response_format: type[ResponseFormatT] | None = None,
    tools: list[Callable] | None = None,
) -> ResponseFormatT | str:
    client = Agent(tools=tools)
    if instruction:
        client.add_message(instruction, "system")

    if isinstance(messages, str):
        messages = [messages]

    for message in messages:
        client.add_message(message, "user")

    return client.create(response_format=response_format)


def create_chat(tools: list[Callable] | None = None) -> Agent:
    return Agent(tools=tools)
