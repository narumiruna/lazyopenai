import asyncio
from typing import Annotated

from lazyopenai.async_api import generate


def add_numbers(
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Add two numbers"""
    return a + b


async def main() -> None:
    resp = await generate(
        "100 + 10 = ?",
        tools=[add_numbers],
    )
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
