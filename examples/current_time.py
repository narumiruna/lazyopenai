from datetime import datetime

from pydantic import BaseModel

from lazyopenai import generate


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CurrentTime(BaseModel):
    current_time: str


def main() -> None:
    resp = generate(
        "What is the current time?",
        tools=[get_current_time],
    )
    print(resp)

    resp_obj = generate(
        "What is the current time?",
        response_format=CurrentTime,
        tools=[get_current_time],
    )
    print(resp_obj)


if __name__ == "__main__":
    main()
