from typing import Literal

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from .settings import settings
from .types import LazyTool
from .types import ResponseFormatT


class LazyClient:
    def __init__(self, tools: list[type[LazyTool]] | None = None) -> None:
        self.client = OpenAI(api_key=settings.api_key)
        self.messages: list = []
        self.tools = {tool.__name__: tool for tool in tools} if tools else {}

    def _generate(self, messages, response_format: type[ResponseFormatT] | None = None):
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

    def _process_tool_calls_in_response(
        self, response: ChatCompletion, response_format: type[ResponseFormatT] | None = None
    ):
        if not self.tools:
            return response

        if not response.choices:
            return response

        finish_reason = response.choices[0].finish_reason
        if finish_reason != "tool_calls":
            return response

        response_message = response.choices[0].message
        if not response_message.tool_calls:
            return response
        self.messages += [response_message]

        for tool_call in response_message.tool_calls:
            tool = self.tools.get(tool_call.function.name)
            if not tool:
                continue

            function_result = tool.call(tool_call.function.arguments)
            self.add_tool_message(function_result, tool_call.id)

        return self._generate(self.messages, response_format=response_format)

    def add_message(self, content: str, role: Literal["system", "user", "assistant"] = "user") -> None:
        self.messages += [{"role": role, "content": content}]

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        self.messages += [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            }
        ]

    def generate(self, response_format: type[ResponseFormatT] | None = None) -> ResponseFormatT | str:
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
