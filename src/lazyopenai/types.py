from typing import TypeVar

from pydantic import BaseModel

ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


class LazyTool(BaseModel):
    def __call__(self):
        raise NotImplementedError
