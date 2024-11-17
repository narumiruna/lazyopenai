import functools

from openai import AsyncOpenAI
from openai import OpenAI

from .settings import settings


@functools.cache
def get_client() -> OpenAI:
    """
    Get a cached instance of the OpenAI client.

    Returns:
        OpenAI: An instance of the OpenAI client.
    """
    return OpenAI(api_key=settings.api_key)


@functools.cache
def get_async_client() -> AsyncOpenAI:
    """
    Get a cached instance of the AsyncOpenAI client.

    Returns:
        AsyncOpenAI: An instance of the AsyncOpenAI client.
    """
    return AsyncOpenAI(api_key=settings.api_key)
