from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import cast

from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

from .client import get_openai_client
from .schema import generate_function_schema
from .settings import get_settings

ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


class Agent:
    def __init__(self, tools: list[Callable] | None = None) -> None:
        self._client = get_openai_client()
        self._messages: list[dict[str, Any]] = []
        self._settings = get_settings()

        tools = tools or []
        self._tools = {tool.__name__: tool for tool in tools}
        self._tool_schemas = [generate_function_schema(tool) for tool in tools]

    def _parse(self, response_format: type[ResponseFormatT]) -> ParsedChatCompletion:
        response = self._client.beta.chat.completions.parse(
            messages=self._messages,
            model=self._settings.openai_model,
            temperature=self._settings.openai_temperature,
            tools=self._tool_schemas,
            max_tokens=self._settings.openai_max_tokens,
            response_format=response_format,
        )  # type: ignore
        if not response.choices:
            return response

        self._messages += [response.choices[0].message.model_dump()]
        return response

    def _create(self) -> ChatCompletion:
        response = self._client.chat.completions.create(
            messages=self._messages,
            model=self._settings.openai_model,
            temperature=self._settings.openai_temperature,
            tools=self._tool_schemas,
            max_tokens=self._settings.openai_max_tokens,
        )  # type: ignore

        if not response.choices:
            return response

        self._messages += [response.choices[0].message.model_dump()]
        return response

    def _handle_response(
        self,
        response: ChatCompletion,
        response_format: type[ResponseFormatT] | None = None,
    ) -> ChatCompletion | ParsedChatCompletion:
        if not response.choices:
            return response

        choice = response.choices[0]
        match choice.finish_reason:
            case "tool_calls":
                self._handle_tool_choice(choice)
                if response_format:
                    return self._parse(response_format)
                return self._create()
            case "stop":
                return response
            case _:
                logger.warning("Unhandled finish reason: {}", choice.finish_reason)
                return response

    def _handle_tool_choice(self, choice: Choice) -> None:
        if not choice.message.tool_calls:
            return

        for tool_call in choice.message.tool_calls:
            tool = self._tools.get(tool_call.function.name)
            if not tool:
                logger.warning("Tool not found: {}", tool_call.function.name)
                continue

            result = str(tool(**json.loads(tool_call.function.arguments)))
            self.add_message(result, "tool", tool_call.id)

    def add_message(
        self, content: str, role: Literal["system", "user", "tool"] = "user", tool_call_id: str | None = None
    ) -> None:
        match role:
            case "user" | "system":
                self._messages += [{"content": content, "role": role}]
            case "tool":
                self._messages += [{"content": content, "role": role, "tool_call_id": tool_call_id}]
            case _:
                raise ValueError(f"Invalid role: {role}")

    def send(self) -> str | None:
        response = self._create()
        response = self._handle_response(response)
        if not response.choices:
            return None
        return response.choices[0].message.content

    def parse(self, response_format: type[ResponseFormatT]) -> ResponseFormatT | None:
        response = self._parse(response_format)
        response = cast(ParsedChatCompletion, self._handle_response(response, response_format))

        if not response.choices:
            return None

        return response.choices[0].message.parsed
