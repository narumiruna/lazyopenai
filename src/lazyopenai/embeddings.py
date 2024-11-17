from openai.types import CreateEmbeddingResponse

from .utils import get_client
from .utils import get_embedding_model


def create(texts: str | list[str]) -> CreateEmbeddingResponse:
    """
    Creates embeddings for the given text or list of texts.

    Args:
        texts (str | list[str]): The text or list of texts to create embeddings for.

    Returns:
        CreateEmbeddingResponse: The response containing the created embeddings.

    Raises:
        ValueError: If the input texts are not a string or list of strings.
    """
    if isinstance(texts, str):
        texts = [texts]

    client = get_client()
    model = get_embedding_model()

    response = client.embeddings.create(input=texts, model=model)
    return response
