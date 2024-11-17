from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias
from typing import cast

from openai.types.chat import ChatCompletionMessageParam

Messages: TypeAlias = (
    str | Iterable[str] | dict[str, str] | Iterable[dict[str, str]] | Iterable[ChatCompletionMessageParam]
)


def to_openai_messages(messages: Messages) -> Iterable[ChatCompletionMessageParam]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if not isinstance(messages, Iterable):
        raise ValueError("Messages must be a string, a dictionary, or an iterable of those types")

    outputs: list[ChatCompletionMessageParam] = []
    for message in messages:
        if isinstance(message, str):
            outputs.append({"role": "user", "content": message})
        elif isinstance(message, dict) and "role" in message and "content" in message:
            outputs.append(cast(ChatCompletionMessageParam, message))
        else:
            raise ValueError(
                "Each message must be a string, a dictionary with 'role' and 'content' keys, "
                "or an iterable of those types"
            )

    return outputs
