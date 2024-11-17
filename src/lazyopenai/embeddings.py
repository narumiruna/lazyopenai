from openai.types import CreateEmbeddingResponse

from .settings import settings
from .utils import get_client


def create(texts: str | list[str]) -> CreateEmbeddingResponse:
    if isinstance(texts, str):
        texts = [texts]

    client = get_client()

    response = client.embeddings.create(input=texts, model=settings.embedding_model)
    return response
