import json
from typing import Literal
from typing import TypeVar

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from .settings import settings
from .types import LazyTool

T = TypeVar("T", bound=BaseModel)


class LazyClient:
    def __init__(self, tools: list[type[LazyTool]] | None = None) -> None:
        self.client = OpenAI(api_key=settings.api_key)
        self.messages: list = []
        self.tools = {tool.__name__: tool for tool in tools} if tools else None

    def _create(self, messages) -> ChatCompletion:
        if self.tools:
            return self.client.chat.completions.create(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
                tools=[openai.pydantic_function_tool(tool) for tool in self.tools.values()],
            )
        else:
            return self.client.chat.completions.create(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
            )

    def _parse(self, messages, response_format: type[T]):
        if self.tools:
            return self.client.beta.chat.completions.parse(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
                tools=[openai.pydantic_function_tool(tool) for tool in self.tools.values()],
                response_format=response_format,
            )
        else:
            return self.client.beta.chat.completions.parse(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
                response_format=response_format,
            )

    def _handle_tool_calls(self, response: ChatCompletion, response_format: type[T] | None = None):
        if not self.tools:
            return response

        if not response.choices:
            return response

        choice = response.choices[0]
        self.messages += [choice.message]

        if choice.finish_reason != "tool_calls":
            return response

        if not choice.message.tool_calls:
            return response

        for tool_call in choice.message.tool_calls:
            tool = self.tools.get(tool_call.function.name)
            if not tool:
                continue

            tool_args = json.loads(tool_call.function.arguments)
            self.messages += [
                {
                    "role": "tool",
                    "content": str(tool(**tool_args)()),
                    "tool_call_id": tool_call.id,
                }
            ]

        if response_format:
            return self._parse(self.messages, response_format)
        else:
            return self._create(self.messages)

    def add_message(self, content: str, role: Literal["system", "user", "assistant"] = "user") -> None:
        self.messages += [{"role": role, "content": content}]

    def create(self) -> str:
        response = self._create(self.messages)
        response = self._handle_tool_calls(response)

        if not response.choices:
            raise ValueError("No completion choices returned")

        content = response.choices[0].message.content
        if not content:
            raise ValueError("No completion message content")

        return content

    def parse(self, response_format: type[T]) -> T:
        response = self._parse(self.messages, response_format)
        response = self._handle_tool_calls(response, response_format)

        if not response.choices:
            raise ValueError("No completion choices returned")

        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("No completion message parsed")

        return parsed
