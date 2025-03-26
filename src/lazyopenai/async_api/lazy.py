from collections.abc import Callable

from .agent import Agent
from .agent import ResponseFormatT


async def generate(
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

    response = await client.create(response_format=response_format)
    return response


def create_async_agent(tools: list[Callable] | None = None) -> Agent:
    return Agent(tools=tools)
