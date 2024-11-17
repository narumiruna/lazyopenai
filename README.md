# Lazy OpenAI

## Installation

```sh
pip install lazyopenai
```

## Usage

```python
from lazyopenai import generate_text

print(generate_text("Hi"))
```

### Structured Outputs

```python
from pydantic import BaseModel
from rich import print

from lazyopenai import generate_object


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


# https://platform.openai.com/docs/guides/structured-outputs?context=ex1#chain-of-thought
resp = generate_object("how can I solve 8x + 7 = -23", MathReasoning)
print(resp)
```
