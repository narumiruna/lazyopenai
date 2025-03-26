from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import TypeVar

from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

from .client import get_openai_client
from .function_schema import generate_function_schema
from .settings import get_settings

ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


class Agent:
    def __init__(self, tools: list[Callable] | None = None) -> None:
        logger.debug("Initializing Chat with tools: {}", tools)
        self.client = get_openai_client()
        self.messages: list[dict[str, Any]] = []
        self.tools = {tool.__name__: tool for tool in tools} if tools else {}
        self.settings = get_settings()

    def _create(self, response_format: type[ResponseFormatT] | None = None) -> ChatCompletion | ParsedChatCompletion:
        logger.debug("Creating chat completion")

        kwargs = {
            "messages": self.messages,
            "model": self.settings.openai_model,
            "temperature": self.settings.openai_temperature,
        }
        if self.tools:
            logger.info("tools: {}", self.tools)
            kwargs["tools"] = [generate_function_schema(tool) for tool in self.tools.values()]

        if response_format:
            logger.info("response_format: {}", response_format)
            kwargs["response_format"] = response_format

        if self.settings.openai_max_tokens:
            kwargs["max_tokens"] = self.settings.openai_max_tokens

        response: ChatCompletion | ParsedChatCompletion
        if response_format:
            response = self.client.beta.chat.completions.parse(**kwargs)  # type: ignore
        else:
            response = self.client.chat.completions.create(**kwargs)  # type: ignore

        logger.debug("Chat completion created: {}", response)

        if not response.choices:
            return response

        self.messages += [response.choices[0].message.model_dump()]
        return response

    def _handle_response(
        self,
        response: ChatCompletion | ParsedChatCompletion,
        response_format: type[ResponseFormatT] | None = None,
    ):
        logger.debug("Handling response")
        if not self.tools:
            return response

        if not response.choices:
            return response

        finish_reason = response.choices[0].finish_reason
        logger.debug("Finish reason: {}", finish_reason)
        if finish_reason != "tool_calls":
            return response

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return response

        for tool_call in tool_calls:
            tool = self.tools.get(tool_call.function.name)
            if not tool:
                continue

            logger.debug("Calling tool: {}", tool_call.function.name)

            function_result = tool(**json.loads(tool_call.function.arguments))
            self.add_message(str(function_result), "tool", tool_call.id)

        return self._create(response_format=response_format)

    def add_message(
        self, content: str, role: Literal["system", "user", "tool"] = "user", tool_call_id: str | None = None
    ) -> None:
        logger.debug("Adding message with content: {} and role: {}", content, role)
        match role:
            case "user" | "system":
                self.messages += [{"content": content, "role": role}]
            case "tool":
                self.messages += [{"content": content, "role": role, "tool_call_id": tool_call_id}]
            case _:
                raise ValueError(f"Invalid role: {role}")

    def create(self, response_format: type[ResponseFormatT] | None = None) -> ResponseFormatT | str:
        logger.debug("Creating final response")
        response = self._handle_response(self._create(response_format), response_format)
        if not response.choices:
            raise ValueError("No completion choices returned")

        response_message = response.choices[0].message
        if response_format:
            logger.info("response_format: {}", response_format)
            if not response_message.parsed:
                raise ValueError("No completion parsed content returned")
            return response_message.parsed

        if not response_message.content:
            raise ValueError("No completion content returned")
        return response_message.content
