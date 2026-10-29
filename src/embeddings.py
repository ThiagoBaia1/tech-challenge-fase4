from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME


class EmbeddingsManager:
    """Gerencia a geração de representações vetoriais de textos."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"[*] Carregando modelo de embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """Gera vetores para uma lista de textos (documentos da base)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalizar vetores facilita cálculo por cosseno
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Gera o vetor para a pergunta do usuário."""
        embedding = self.model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.tolist()