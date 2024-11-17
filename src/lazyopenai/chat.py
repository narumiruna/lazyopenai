import json
from typing import TypeVar

import openai
from pydantic import BaseModel

from .settings import settings
from .utils import get_async_client
from .utils import get_client

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=BaseModel)


def create(messages, tools: list[type[S]] | None = None) -> str:
    client = get_client()

    tool_dict = {tool.__name__: tool for tool in tools} if tools else {}

    response = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        tools=[openai.pydantic_function_tool(tool) for tool in tools] if tools else None,
    )

    # handle tool calls
    if tools and response.choices:
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            tool_messages = []
            for tool_call in choice.message.tool_calls:
                tool_function = tool_dict.get(tool_call.function.name)
                if not tool_function:
                    continue
                tool_arguments = json.loads(tool_call.function.arguments)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_function(**tool_arguments).call()),
                        "tool_call_id": tool_call.id,
                    }
                )
            response = client.chat.completions.create(
                messages=messages + [choice.message] + tool_messages,
                model=settings.model,
            )

    if not response.choices:
        raise ValueError("No completion choices returned")

    content = response.choices[0].message.content
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


def parse(messages, response_format: type[T], tools: list[type[S]] | None = None) -> T:
    client = get_client()

    tool_dict = {tool.__name__: tool for tool in tools} if tools else {}

    response = client.beta.chat.completions.parse(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        response_format=response_format,
        tools=[openai.pydantic_function_tool(tool) for tool in tools] if tools else None,
    )

    # handle tool calls
    if tools and response.choices:
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            tool_messages = []
            for tool_call in choice.message.tool_calls:
                tool_function = tool_dict.get(tool_call.function.name)
                if not tool_function:
                    continue
                tool_arguments = json.loads(tool_call.function.arguments)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_function(**tool_arguments).call()),
                        "tool_call_id": tool_call.id,
                    }
                )
            response = client.beta.chat.completions.parse(
                model=settings.model,
                messages=messages + [choice.message] + tool_messages,
                temperature=settings.temperature,
                response_format=response_format,
            )

    if not response.choices:
        raise ValueError("No completion choices returned")

    parsed = response.choices[0].message.parsed
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
