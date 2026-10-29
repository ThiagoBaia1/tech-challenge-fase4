from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Arquivos do índice FAISS
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = PROCESSED_DATA_DIR / "faiss_metadata.parquet"

# Configuração do modelo de embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 64

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)