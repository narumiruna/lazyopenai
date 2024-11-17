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

    kwargs = {}
    tool_dict = {}
    if tools:
        tool_dict = {tool.__name__: tool for tool in tools}
        kwargs["tools"] = [openai.pydantic_function_tool(tool) for tool in tools]
    response = client.chat.completions.create(
        messages=messages,
        model=settings.model,
        temperature=settings.temperature,
        **kwargs,  # type: ignore
    )

    # handle tool calls
    if tools and response.choices:
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_messages = []
            for tool_call in choice.message.tool_calls:
                tool_function = tool_dict.get(tool_call.function.name)
                if not tool_function:
                    continue
                tool_arguments = json.loads(tool_call.function.arguments)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_function(**tool_arguments)()),  # type: ignore
                        "tool_call_id": tool_call.id,
                    }
                )
            response = client.chat.completions.create(
                messages=messages + [choice.message] + tool_messages,
                model=settings.model,
                temperature=settings.temperature,
            )

    if not response.choices:
        raise ValueError("No completion choices returned")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content


async def async_create(messages) -> str:
    client = get_async_client()

    response = await client.chat.completions.create(
        messages=messages,
        model=settings.model,
        temperature=settings.temperature,
    )

    if not response.choices:
        raise ValueError("No completion choices returned")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("No completion message content")

    return content


def parse(messages, response_format: type[T], tools: list[type[S]] | None = None) -> T:
    client = get_client()

    openai_tools = None
    if tools:
        openai_tools = []
        for tool in tools:
            openai_tools.append(openai.pydantic_function_tool(tool))

    kwargs = {}
    tool_dict = {}
    if tools:
        tool_dict = {tool.__name__: tool for tool in tools}
        kwargs["tools"] = [openai.pydantic_function_tool(tool) for tool in tools]

    response = client.beta.chat.completions.parse(
        messages=messages,
        model=settings.model,
        temperature=settings.temperature,
        response_format=response_format,
        **kwargs,  # type: ignore
    )

    # handle tool calls
    if tools and response.choices:
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_messages = []
            for tool_call in choice.message.tool_calls:
                tool_function = tool_dict.get(tool_call.function.name)
                if not tool_function:
                    continue
                tool_arguments = json.loads(tool_call.function.arguments)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_function(**tool_arguments)()),  # type: ignore
                        "tool_call_id": tool_call.id,
                    }
                )
            response = client.beta.chat.completions.parse(
                messages=messages + [choice.message] + tool_messages,
                model=settings.model,
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

    response = await client.beta.chat.completions.parse(
        messages=messages,
        model=settings.model,
        temperature=settings.temperature,
        response_format=response_format,
    )

    if not response.choices:
        raise ValueError("No completion choices returned")

    parsed = response.choices[0].message.parsed
    if not parsed:
        raise ValueError("No completion message parsed")

    return parsed
