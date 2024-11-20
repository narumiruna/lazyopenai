from pydantic import Field

from lazyopenai import generate
from lazyopenai.types import BaseTool


class AddNumbers(BaseTool):
    a: float = Field(..., description="First number to add")
    b: float = Field(..., description="Second number to add")

    def __call__(self) -> float:
        print("function called")
        return self.a + self.b


def main() -> None:
    resp = generate(
        "100 + 10 = ?",
        tools=[AddNumbers],
    )
    print(resp)


if __name__ == "__main__":
    main()
