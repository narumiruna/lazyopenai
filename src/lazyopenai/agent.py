from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal
from typing import TypeVar

from loguru import logger
from openai.types.responses import ParsedResponse
from openai.types.responses import ParsedResponseFunctionToolCall
from openai.types.responses import Response
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel

from .client import get_openai_client
from .schema import generate_function_schema
from .settings import get_settings

TextFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


class Agent:
    def __init__(self, tools: list[Callable] | None = None) -> None:
        self._client = get_openai_client()
        self._messages: ResponseInputParam = []
        self._settings = get_settings()

        tools = tools or []
        self._tools = {tool.__name__: tool for tool in tools}
        self._tool_schemas = [generate_function_schema(tool) for tool in tools]

    def _create(self, text_format: type[TextFormatT] | None = None) -> Response | ParsedResponse:
        if text_format:
            response = self._client.responses.parse(
                input=self._messages,
                model=self._settings.openai_model,
                text_format=text_format,
                temperature=self._settings.openai_temperature,
                tools=self._tool_schemas,
                max_output_tokens=self._settings.openai_max_tokens,
            )
        else:
            response = self._client.responses.create(
                input=self._messages,
                model=self._settings.openai_model,
                temperature=self._settings.openai_temperature,
                tools=self._tool_schemas,
                max_output_tokens=self._settings.openai_max_tokens,
            )

        for output_item in response.output:
            data = output_item.model_dump()
            # handle 'Unknown parameter' error
            if isinstance(output_item, ParsedResponseFunctionToolCall):
                del data["parsed_arguments"]
            self._messages += [data]
        return response

    def _handle_tool_response(
        self,
        response: Response | ParsedResponse,
        text_format: type[TextFormatT] | None = None,
    ) -> Response | ParsedResponse:
        function_call_outputs = []
        for tool_call in response.output:
            if not isinstance(tool_call, ResponseFunctionToolCall):
                continue

            tool = self._tools.get(tool_call.name)
            if not tool:
                logger.warning("Tool not found: {}", tool_call.name)
                continue

            result = str(tool(**json.loads(tool_call.arguments)))
            function_call_outputs += [
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result),
                }
            ]
        if function_call_outputs:
            self._messages += function_call_outputs
        response = self._create(text_format)
        return response

    def add_message(self, content: str, role: Literal["system", "user"] = "user") -> None:
        self._messages += [{"content": content, "role": role}]

    def create(self, text_format: type[TextFormatT] | None = None) -> TextFormatT | str:
        response = self._create(text_format)
        response = self._handle_tool_response(response, text_format)

        if isinstance(response, ParsedResponse):
            parsed = response.output_parsed
            if parsed:
                return parsed

        return response.output_text
