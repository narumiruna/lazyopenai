import json
from typing import TypeVar

from pydantic import BaseModel

ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


class LazyTool(BaseModel):
    def __call__(self):
        raise NotImplementedError

    @classmethod
    def call(cls, arguments: str) -> str:
        try:
            func_args = json.loads(arguments)
        except json.JSONDecodeError:
            return "failed to decode function arguments"
        return str(cls(**func_args)())
