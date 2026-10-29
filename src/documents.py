import re
from typing import Any, Dict, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class ReviewDocument:
    id: str
    page_content: str
    metadata: Dict[str, Any]


def clean_text(text: str) -> str:
    """Limpeza leve preservando pontuação semântica e acentuação."""
    if not isinstance(text, str):
        return ""

    # Converte quebras de linha em espaço simples
    text = re.sub(r"[\r\n]+", " ", text)
    # Remove múltiplos espaços
    text = re.sub(r"\s+", " ", text)
    # Suprime pontuações excessivamente repetidas (ex: "!!!!" -> "!")
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    return text.strip()


def build_review_text(title: Any, message: Any) -> str:
    """Combina título e mensagem de forma natural."""
    clean_title = clean_text(str(title)) if pd.notna(title) else ""
    clean_msg = clean_text(str(message)) if pd.notna(message) else ""

    if clean_title and clean_msg:
        # Garante pontuação entre título e corpo
        separator = " - " if not clean_title.endswith((".", "!", "?")) else " "
        return f"{clean_title}{separator}{clean_msg}"
    return clean_title or clean_msg


def create_documents_from_df(df: pd.DataFrame) -> List[ReviewDocument]:
    """Converte o DataFrame processado em uma lista estruturada de ReviewDocument."""
    documents = []

    for _, row in df.iterrows():
        content = build_review_text(
            row.get("review_comment_title"),
            row.get("review_comment_message"),
        )

        if not content:
            continue

        doc = ReviewDocument(
            id=str(row["review_id"]),
            page_content=content,
            metadata={
                "review_id": str(row["review_id"]),
                "order_id": str(row["order_id"]),
                "score": int(row["review_score"]),
                "category": (
                    str(row["product_category_name"])
                    if pd.notna(row.get("product_category_name"))
                    else "desconhecido"
                ),
                "is_delayed": bool(row.get("is_delayed", False)),
                "created_at": (
                    str(row.get("review_creation_date"))
                    if pd.notna(row.get("review_creation_date"))
                    else ""
                ),
            },
        )
        documents.append(doc)

    return documents