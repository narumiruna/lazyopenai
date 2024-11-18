from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from ..settings import settings


def create_embeddings(texts: str | list[str]) -> CreateEmbeddingResponse:
    if isinstance(texts, str):
        texts = [texts]

    client = OpenAI(api_key=settings.api_key)

    response = client.embeddings.create(input=texts, model=settings.embedding_model)
    return response
