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
        self.tools = {tool.__name__: tool for tool in tools} if tools else {}

    def _generate(self, messages, response_format: type[T] | None = None):
        kwargs = {
            "messages": messages,
            "model": settings.model,
            "temperature": settings.temperature,
        }
        if self.tools:
            kwargs["tools"] = [openai.pydantic_function_tool(tool) for tool in self.tools.values()]
        if response_format:
            kwargs["response_format"] = response_format

        if response_format:
            return self.client.beta.chat.completions.parse(**kwargs)
        else:
            return self.client.chat.completions.create(**kwargs)

    def _process_tool_calls_in_response(self, response: ChatCompletion, response_format: type[T] | None = None):
        if not self.tools or not response.choices:
            return response

        choice = response.choices[0]
        self.messages += [choice.message]

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
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

        return self._generate(self.messages, response_format=response_format)

    def add_message(self, content: str, role: Literal["system", "user", "assistant"] = "user"):
        self.messages += [{"role": role, "content": content}]

    def generate(self, response_format: type[T] | None = None) -> T | str:
        response = self._process_tool_calls_in_response(self._generate(self.messages, response_format), response_format)
        if not response.choices:
            raise ValueError("No completion choices returned")

        response_message = response.choices[0].message

        if response_format:
            if not response_message.parsed:
                raise ValueError("No completion parsed content returned")
            return response_message.parsed

        if not response_message.content:
            raise ValueError("No completion content returned")

        return response_message.content
