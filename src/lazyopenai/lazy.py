from collections.abc import Callable
from typing import cast

from .agent import Agent
from .agent import TextFormatT


def generate(
    messages: str | list[str],
    instruction: str | None = None,
    response_format: type[TextFormatT] | None = None,
    tools: list[Callable] | None = None,
) -> TextFormatT | str:
    client = Agent(tools=tools)
    if instruction:
        client.add_message(instruction, "system")

    if isinstance(messages, str):
        messages = [messages]

    for message in messages:
        client.add_message(message, "user")

    return client.create(text_format=response_format)


def send(
    messages: str | list[str],
    instruction: str | None = None,
    tools: list[Callable] | None = None,
) -> str:
    return generate(
        messages=messages,
        instruction=instruction,
        tools=tools,
    )


def parse(
    messages: str | list[str],
    response_format: type[TextFormatT],
    instruction: str | None = None,
    tools: list[Callable] | None = None,
) -> TextFormatT:
    response = generate(
        messages=messages,
        instruction=instruction,
        response_format=response_format,
        tools=tools,
    )
    return cast(TextFormatT, response)


def create_agent(tools: list[Callable] | None = None) -> Agent:
    return Agent(tools=tools)
