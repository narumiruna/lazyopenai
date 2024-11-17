from typing import TypeVar

from pydantic import BaseModel

from .chat import create
from .chat import parse

T = TypeVar("T", bound=BaseModel)


def generate_text(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return create(messages)


def generate_object(prompt: str, response_format: type[T], system: str | None = None) -> T:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return parse(messages, response_format)
