import json

from pydantic import BaseModel
from pydantic import Field

from lazyopenai import generate


# classifies a URL and returns whether it contains a video
class URL(BaseModel):
    url: str = Field(..., description="The URL")
    contain_video: bool = Field(..., description="Whether the URL contains a video")

    def __call__(self) -> str:
        return json.dumps({"url": self.url, "contain_video": self.contain_video})


def main() -> None:
    urls = [
        "https://www.google.com.tw",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
    ]

    for url in urls:
        resp = generate(
            f"Classify the URL and return whether it contains a video: {url}",
            tools=[URL],
        )
        print(resp)


if __name__ == "__main__":
    main()
