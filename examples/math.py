from pydantic import BaseModel
from rich import print

from lazyopenai import generate_object


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


def main() -> None:
    # https://platform.openai.com/docs/guides/structured-outputs?context=ex1#chain-of-thought
    resp = generate_object("how can I solve 8x + 7 = -23", MathReasoning)
    print(resp)


if __name__ == "__main__":
    main()
