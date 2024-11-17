from typing import TypeVar

from pydantic import BaseModel

from .chat import create
from .chat import parse

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=BaseModel)


def generate(
    user: str,
    system: str | None = None,
    response_format: type[T] | None = None,
    tools: list[type[S]] | None = None,
) -> T | str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    if response_format:
        return parse(messages, response_format, tools=tools)

    return create(messages, tools=tools)
