from typing import TypeVar

from pydantic import BaseModel

from .settings import settings
from .utils import get_async_client
from .utils import get_client

T = TypeVar("T", bound=BaseModel)


def create(messages) -> str:
    client = get_client()

    completion = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content


async def async_create(messages) -> str:
    client = get_async_client()

    completion = await client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content


def parse(messages, response_format: type[T]) -> T:
    client = get_client()

    completion = client.beta.chat.completions.parse(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        response_format=response_format,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    parsed = completion.choices[0].message.parsed
    if not parsed:
        raise ValueError("No completion message parsed")

    return parsed


async def async_parse(messages, response_format: type[T]) -> T:
    client = get_async_client()

    completion = await client.beta.chat.completions.parse(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        response_format=response_format,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    parsed = completion.choices[0].message.parsed
    if not parsed:
        raise ValueError("No completion message parsed")

    return parsed
