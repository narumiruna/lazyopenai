import functools
import os
from typing import Final

from openai import AsyncOpenAI
from openai import OpenAI

DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
DEFAULT_TEMPERATURE: Final[float] = 0.0


@functools.cache
def get_client() -> OpenAI:
    """
    Get a cached instance of the OpenAI client.

    Returns:
        OpenAI: An instance of the OpenAI client.
    """
    return OpenAI()


@functools.cache
def get_async_client() -> AsyncOpenAI:
    """
    Get a cached instance of the AsyncOpenAI client.

    Returns:
        AsyncOpenAI: An instance of the AsyncOpenAI client.
    """
    return AsyncOpenAI()


@functools.cache
def get_model() -> str:
    """
    Get the model name from the environment variable or use the default model.

    Returns:
        str: The model name.
    """
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


@functools.cache
def get_temperature() -> float:
    """
    Retrieves the temperature setting for the OpenAI API from environment variables.
    Uses a default value if the environment variable is not set.

    Returns:
        float: The temperature setting for the OpenAI API.
    """
    return float(os.getenv("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE))


@functools.cache
def get_embedding_model() -> str:
    """
    Retrieves the embedding model setting for the OpenAI API from environment variables.
    Uses a default value if the environment variable is not set.

    Returns:
        str: The embedding model setting for the OpenAI API.
    """
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
