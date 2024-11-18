from pydantic import BaseModel
from pydantic import Field

from lazyopenai import generate


class URL(BaseModel):
    url: str = Field(..., description="The URL")
    contain_video: bool = Field(..., description="Whether the URL contains a video")


class URLs(BaseModel):
    urls: list[URL] = Field(..., description="The list of URLs")


def main() -> None:
    prompt = """
    https://www.google.com.tw
    https://www.youtube.com/watch?v=9bZkp7q19f0
    """.strip()

    resp = generate(prompt, response_format=URLs)
    print(resp)


if __name__ == "__main__":
    main()
