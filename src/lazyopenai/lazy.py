from typing import TypeVar

from pydantic import BaseModel

from .chat import create
from .chat import parse

T = TypeVar("T", bound=BaseModel)


def generate_text(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return create(messages)


def generate_object(prompt: str, response_format: type[T]) -> T:
    messages = [{"role": "user", "content": prompt}]
    return parse(messages, response_format)
