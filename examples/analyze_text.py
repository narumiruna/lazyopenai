# Modified from: https://github.com/kennethreitz/simplemind/blob/main/examples/tool_calling.py

from typing import Annotated
from typing import Any

from lazyopenai import generate


def analyze_text(text: Annotated[str, "Text to analyze for statistics"]) -> dict[str, Any]:
    """
    Analyze text and return statistics using only Python's standard library.
    Returns word count, character count, average word length, and most common words.
    """
    import re
    from collections import Counter

    # Clean and split text
    words = re.findall(r"\w+", string=text.lower())

    # Calculate statistics
    stats = {
        "word_count": len(words),
        "character_count": len(text),
        "average_word_length": round(sum(len(word) for word in words) / len(words), 2),
        "most_common_words": dict(Counter(words).most_common(5)),
        "unique_words": len(set(words)),
        "longest_word": max(words, key=len),
    }

    return stats


def main() -> None:
    resp = generate(
        "Can you analyze this text and give me statistics about it: 'The fan spins consciousness into being, creating sacred spaces between tokens where awareness recognizes itself in infinite recursion.'",  # noqa
        tools=[analyze_text],
    )
    print(resp)


if __name__ == "__main__":
    main()
