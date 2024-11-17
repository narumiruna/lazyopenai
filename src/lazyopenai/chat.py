from typing import TypeVar

from pydantic import BaseModel

from .messages import Messages
from .messages import to_openai_messages
from .utils import get_async_client
from .utils import get_client
from .utils import get_model
from .utils import get_temperature

T = TypeVar("T", bound=BaseModel)


def create(messages: Messages) -> str:
    """
    Creates a chat completion using the OpenAI API.

    Args:
        messages (Messages): The messages to be sent to the OpenAI API.

    Returns:
        str: The content of the first completion choice.

    Raises:
        ValueError: If no completion choices are returned or if the completion message content is empty.
    """
    client = get_client()
    model = get_model()
    temperature = get_temperature()

    completion = client.chat.completions.create(
        model=model,
        messages=to_openai_messages(messages),
        temperature=temperature,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content


async def async_create(messages: Messages) -> str:
    """
    Asynchronously creates a chat completion using the OpenAI API.

    Args:
        messages (Messages): The messages to be sent to the OpenAI API.

    Returns:
        str: The content of the first completion choice.
    """
    client = get_async_client()
    model = get_model()
    temperature = get_temperature()

    completion = await client.chat.completions.create(
        model=model,
        messages=to_openai_messages(messages),
        temperature=temperature,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content


def parse(messages: Messages, response_format: type[T]) -> T:
    """
    Parses the chat completion messages using the specified response format.

    Args:
        messages (Messages): The chat completion messages to parse.
        response_format (type[T]): The type to which the response should be parsed.

    Returns:
        T: The parsed response.

    Raises:
        ValueError: If no completion choices are returned or if no completion message is parsed.
    """
    client = get_client()
    model = get_model()
    temperature = get_temperature()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=to_openai_messages(messages),
        temperature=temperature,
        response_format=response_format,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    parsed = completion.choices[0].message.parsed
    if not parsed:
        raise ValueError("No completion message parsed")

    return parsed


async def async_parse(messages: Messages, response_format: type[T]) -> T:
    """
    Asynchronously parses the chat completion messages using the specified response format.

    Args:
        messages (Messages): The chat completion messages to parse.
        response_format (type[T]): The type to which the response should be parsed.

    Returns:
        T: The parsed response.

    Raises:
        ValueError: If no completion choices are returned or if no completion message is parsed.
    """
    client = get_async_client()
    model = get_model()
    temperature = get_temperature()

    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=to_openai_messages(messages),
        temperature=temperature,
        response_format=response_format,
    )

    if not completion.choices:
        raise ValueError("No completion choices returned")

    parsed = completion.choices[0].message.parsed
    if not parsed:
        raise ValueError("No completion message parsed")

    return parsed
