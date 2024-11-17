from typing import TypeVar

from pydantic import BaseModel

from .client import LazyClient

T = TypeVar("T", bound=BaseModel)


def generate(
    user: str,
    system: str | None = None,
    response_format: type[T] | None = None,
    tools: list[type[BaseModel]] | None = None,
) -> T | str:
    client = LazyClient(tools=tools)
    if system:
        client.add_message(system, role="system")
    client.add_message(user, role="user")

    if response_format:
        return client.parse(response_format)

    return client.create()
