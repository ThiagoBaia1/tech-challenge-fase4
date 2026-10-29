from pathlib import Path
import sys
import time

# Garante resolução da raiz do projeto no sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR
from src.ingestion import load_raw_data
from src.documents import create_documents_from_df
from src.vectorstore import VectorStoreManager


def main():
    print("=" * 60)
    print("  ETAPA 1 & 2: Construção da Base de Conhecimento e Indexação FAISS")
    print("=" * 60)

    start_total = time.time()

    # 1. Ingestão e tratamento dos dados
    t0 = time.time()
    print("\n[1/3] Carregando e tratando dados brutos da Olist...")
    df = load_raw_data(DATA_DIR)
    print(f"[+] {len(df):,} avaliações válidas carregadas em {time.time() - t0:.2f}s.")

    # 2. Conversão em Documentos Estruturados
    t1 = time.time()
    print("\n[2/3] Convertendo registros em documentos com metadados...")
    docs = create_documents_from_df(df)
    print(f"[+] {len(docs):,} documentos preparados em {time.time() - t1:.2f}s.")

    # 3. Geração de Embeddings e Gravação no FAISS
    t2 = time.time()
    print("\n[3/3] Gerando embeddings e construindo índice vetorial...")
    store = VectorStoreManager()
    store.add_documents(docs, batch_size=128)
    print(f"[+] Indexação vetorial concluída em {time.time() - t2:.2f}s.")

    print("\n" + "=" * 60)
    print(f"[✓] Processo finalizado com sucesso em {time.time() - start_total:.2f}s!")
    print(f"    - Índice binário salvo em: {store.index_path}")
    print(f"    - Metadados salvos em: {store.metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()