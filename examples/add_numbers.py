from typing import Annotated

from lazyopenai import generate


def add_numbers(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers"""
    return a + b


def main() -> None:
    resp = generate(
        "100 + 10 = ?",
        tools=[add_numbers],
    )
    print(resp)


if __name__ == "__main__":
    main()
