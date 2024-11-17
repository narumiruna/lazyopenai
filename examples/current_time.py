from datetime import datetime

from lazyopenai import generate
from lazyopenai.types import LazyTool


class GetCurrentTime(LazyTool):
    def __call__(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CurrentTime(LazyTool):
    current_time: str


def main() -> None:
    resp = generate(
        "What is the current time?",
        tools=[GetCurrentTime],
    )
    print(resp)

    resp_obj = generate(
        "What is the current time?",
        response_format=CurrentTime,
        tools=[GetCurrentTime],
    )
    print(resp_obj)


if __name__ == "__main__":
    main()
