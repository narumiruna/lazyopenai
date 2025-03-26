from collections.abc import Callable

from .agent import Agent
from .agent import ResponseFormatT


def generate(
    messages: str | list[str],
    instruction: str | None = None,
    response_format: type[ResponseFormatT] | None = None,
    tools: list[Callable] | None = None,
) -> ResponseFormatT | str | None:
    client = Agent(tools=tools)
    if instruction:
        client.add_message(instruction, "system")

    if isinstance(messages, str):
        messages = [messages]

    for message in messages:
        client.add_message(message, "user")

    if response_format:
        return client.parse(response_format=response_format)
    return client.send()


def send(
    messages: str | list[str],
    instruction: str | None = None,
    tools: list[Callable] | None = None,
) -> str | None:
    client = Agent(tools=tools)
    if instruction:
        client.add_message(instruction, "system")

    if isinstance(messages, str):
        messages = [messages]

    for message in messages:
        client.add_message(message, "user")

    return client.send()


def create_agent(tools: list[Callable] | None = None) -> Agent:
    return Agent(tools=tools)
