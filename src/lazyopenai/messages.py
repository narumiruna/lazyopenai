from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionFunctionMessageParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionUserMessageParam

Message: TypeAlias = str | ChatCompletionMessageParam
Messages: TypeAlias = Message | Iterable[Message]


def to_openai_messages(messages: Messages) -> Iterable[ChatCompletionMessageParam]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if isinstance(
        messages,
        ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
        | ChatCompletionFunctionMessageParam,
    ):
        return [messages]

    if not isinstance(messages, Iterable):
        raise ValueError("Messages must be a string, ChatCompletionMessageParam, or an iterable of those types")

    outputs: list[ChatCompletionMessageParam] = []
    for message in messages:
        if isinstance(message, str):
            outputs.append({"role": "user", "content": message})
        elif isinstance(
            message,
            ChatCompletionSystemMessageParam
            | ChatCompletionUserMessageParam
            | ChatCompletionAssistantMessageParam
            | ChatCompletionToolMessageParam
            | ChatCompletionFunctionMessageParam,
        ):
            outputs.append(message)
        else:
            raise ValueError("Messages must be a string, ChatCompletionMessageParam, or an iterable of those types")
    return outputs
