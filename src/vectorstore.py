# src/vectorstore.py
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from src.documents import ReviewDocument
from src.embeddings import EmbeddingsManager


class VectorStoreManager:
    """Gerencia a persistência, indexação e busca semântica utilizando FAISS."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
        embeddings_manager: Optional[EmbeddingsManager] = None
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        
        self.index: Optional[faiss.Index] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        
        # Carrega o índice do disco se já existir
        self.load_index()

    def load_index(self) -> bool:
        """Carrega o índice FAISS e a tabela de metadados salvos."""
        if self.index_path.exists() and self.metadata_path.exists():
            print(f"[*] Carregando índice FAISS de: {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            self.metadata_df = pd.read_parquet(self.metadata_path)
            print(f"[+] Índice carregado! Total de registros: {self.index.ntotal}")
            return True
        return False

    def add_documents(self, documents: List[ReviewDocument], batch_size: int = 128) -> None:
        """Calcula os embeddings e constrói o índice FAISS."""
        total_docs = len(documents)
        print(f"[*] Gerando embeddings para {total_docs:,} documentos...")

        texts = [doc.page_content for doc in documents]
        
        # 1. Gera matriz de embeddings em batch
        raw_embeddings = self.embeddings_manager.embed_documents(texts, batch_size=batch_size)
        vectors = np.array(raw_embeddings, dtype=np.float32)

        # 2. Inicializa o índice FAISS (Cosine Similarity via Inner Product com vetores normalizados)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        # 3. Estrutura os metadados associados por índice de linha
        records = []
        for doc in documents:
            record = {"page_content": doc.page_content}
            record.update(doc.metadata)
            records.append(record)
            
        self.metadata_df = pd.DataFrame(records)

        # 4. Grava diretamente em disco de forma síncrona e segura
        print(f"[*] Gravando índice FAISS em: {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_df.to_parquet(self.metadata_path, index=False)
        print(f"[+] Indexação concluída com sucesso! Total gravado: {self.index.ntotal}")

    def search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Any]]:
        """Realiza a busca vetorial por similaridade de cosseno com suporte a filtros."""
        if self.index is None or self.metadata_df is None:
            raise ValueError("O índice FAISS não está carregado. Execute a indexação primeiro.")

        # 1. Vetoriza a consulta
        query_vector = np.array([self.embeddings_manager.embed_query(query)], dtype=np.float32)

        # Se houver filtro de metadados, busca uma amostra maior para filtrar em memória
        fetch_k = k * 10 if filter_metadata else k
        fetch_k = min(fetch_k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, fetch_k)

        out_docs = []
        out_metas = []
        out_distances = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            row = self.metadata_df.iloc[idx].to_dict()
            page_content = row.pop("page_content")

            # Aplicação de filtros de metadados
            if filter_metadata:
                match = all(row.get(key) == val for key, val in filter_metadata.items())
                if not match:
                    continue

            out_docs.append(page_content)
            out_metas.append(row)
            out_distances.append(float(dist))

            if len(out_docs) == k:
                break

        return {
            "documents": [out_docs],
            "metadatas": [out_metas],
            "distances": [out_distances]
        }