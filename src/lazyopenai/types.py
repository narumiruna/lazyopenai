from pydantic import BaseModel


class LazyTool(BaseModel):
    def __call__(self):
        raise NotImplementedError
