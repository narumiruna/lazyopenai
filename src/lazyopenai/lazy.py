from typing import TypeVar

from pydantic import BaseModel

from .client import LazyClient
from .types import LazyTool

T = TypeVar("T", bound=BaseModel)


def generate(
    user: str,
    system: str | None = None,
    response_format: type[T] | None = None,
    tools: list[type[LazyTool]] | None = None,
) -> T | str:
    client = LazyClient(tools=tools)
    if system:
        client.add_message(system, role="system")
    client.add_message(user, role="user")

    return client.generate(response_format=response_format)
