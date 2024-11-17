from typing import TypeVar

from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel

from .chat import create
from .chat import parse

T = TypeVar("T", bound=BaseModel)


def generate_text(prompt: str) -> str:
    return create([ChatCompletionUserMessageParam(role="user", content=prompt)])


def generate_object(prompt: str, response_format: type[T]) -> T:
    return parse([ChatCompletionUserMessageParam(role="user", content=prompt)], response_format)
