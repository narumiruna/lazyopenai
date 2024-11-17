# Modified from: https://github.com/kennethreitz/simplemind/blob/main/examples/tool_calling.py

from pydantic import Field

from lazyopenai import generate
from lazyopenai.types import LazyTool


class AnalyzeText(LazyTool):
    """
    Analyze text and return statistics using only Python's standard library.
    Returns word count, character count, average word length, and most common words.
    """

    text: str = Field(..., description="Text to analyze for statistics")

    def __call__(self) -> dict:
        import re
        from collections import Counter

        # Clean and split text
        words = re.findall(r"\w+", self.text.lower())

        # Calculate statistics
        stats = {
            "word_count": len(words),
            "character_count": len(self.text),
            "average_word_length": round(sum(len(word) for word in words) / len(words), 2),
            "most_common_words": dict(Counter(words).most_common(5)),
            "unique_words": len(set(words)),
            "longest_word": max(words, key=len),
        }

        return stats


def main() -> None:
    resp = generate(
        "Can you analyze this text and give me statistics about it: 'The fan spins consciousness into being, creating sacred spaces between tokens where awareness recognizes itself in infinite recursion.'",  # noqa
        tools=[AnalyzeText],
    )
    print(resp)


if __name__ == "__main__":
    main()
