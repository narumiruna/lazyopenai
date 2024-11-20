from __future__ import annotations

from typing import Literal
from typing import TypeAlias

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage

from .settings import settings
from .types import BaseTool
from .types import ResponseFormatT

Message: TypeAlias = (
    ChatCompletionMessage
    | ChatCompletionSystemMessageParam
    | ChatCompletionToolMessageParam
    | ChatCompletionUserMessageParam
    | ParsedChatCompletionMessage
)


class LazyClient:
    def __init__(self, tools: list[type[BaseTool]] | None = None) -> None:
        self.client = OpenAI(api_key=settings.api_key)
        self.messages: list[Message] = []
        self.tools = {tool.__name__: tool for tool in tools} if tools else {}

    def _generate(
        self, messages, response_format: type[ResponseFormatT] | None = None
    ) -> ChatCompletion | ParsedChatCompletion:
        kwargs = {
            "messages": messages,
            "model": settings.model,
            "temperature": settings.temperature,
        }
        if self.tools:
            kwargs["tools"] = [openai.pydantic_function_tool(tool) for tool in self.tools.values()]
        if response_format:
            kwargs["response_format"] = response_format

        response: ChatCompletion | ParsedChatCompletion
        if response_format:
            response = self.client.beta.chat.completions.parse(**kwargs)
        else:
            response = self.client.chat.completions.create(**kwargs)

        return response

    def _handle_response(
        self, response: ChatCompletion | ParsedChatCompletion, response_format: type[ResponseFormatT] | None = None
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

    def add_message(self, content: str, role: Literal["system", "user"] = "user") -> None:
        match role:
            case "user":
                self.add_user_message(content)
            case "system":
                self.add_system_message(content)
            case _:
                raise ValueError(f"Invalid role: {role}")

    def add_user_message(self, content: str) -> None:
        self.messages += [
            ChatCompletionUserMessageParam(
                content=content,
                role="user",
            )
        ]

    def add_system_message(self, content: str) -> None:
        self.messages += [
            ChatCompletionSystemMessageParam(
                content=content,
                role="system",
            )
        ]

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        self.messages += [
            ChatCompletionToolMessageParam(
                content=content,
                role="tool",
                tool_call_id=tool_call_id,
            )
        ]

    def generate(self, response_format: type[ResponseFormatT] | None = None) -> ResponseFormatT | str:
        response = self._handle_response(self._generate(self.messages, response_format), response_format)
        if not response.choices:
            raise ValueError("No completion choices returned")

        response_message = response.choices[0].message
        self.messages += [response_message]

        if response_format:
            if not response_message.parsed:
                raise ValueError("No completion parsed content returned")
            return response_message.parsed

        if not response_message.content:
            raise ValueError("No completion content returned")

        return response_message.content
