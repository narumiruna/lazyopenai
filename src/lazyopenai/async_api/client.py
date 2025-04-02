from functools import cache

from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI

from ..settings import get_settings


@cache
def get_async_openai_client() -> AsyncOpenAI:
    settings = get_settings()

    if settings.openai_api_key and settings.azure_openai_api_key:
        raise ValueError("Both OpenAI and Azure OpenAI API keys are set. Please set only one.")

    if settings.azure_openai_endpoint and settings.azure_openai_api_key:
        return AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.openai_api_version,
            api_key=settings.azure_openai_api_key,
        )
    elif settings.openai_api_key:
        return AsyncOpenAI(api_key=settings.openai_api_key)
    else:
        raise ValueError("No OpenAI API key set.")
