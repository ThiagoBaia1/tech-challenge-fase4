from pathlib import Path
import pandas as pd


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """Carrega as tabelas principais e constrói o dataset base com metadados."""
    reviews_path = data_dir / "raw" / "olist_order_reviews_dataset.csv"
    orders_path = data_dir / "raw" / "olist_orders_dataset.csv"
    items_path = data_dir / "raw" / "olist_order_items_dataset.csv"
    products_path = data_dir / "raw" / "olist_products_dataset.csv"

    df_reviews = pd.read_csv(reviews_path)
    df_orders = pd.read_csv(orders_path)
    df_items = pd.read_csv(items_path)
    df_products = pd.read_csv(products_path)

    # 1. Filtrar avaliações que contenham pelo menos título ou mensagem
    has_text = (
        df_reviews["review_comment_message"].notna()
        & (df_reviews["review_comment_message"].str.strip() != "")
    ) | (
        df_reviews["review_comment_title"].notna()
        & (df_reviews["review_comment_title"].str.strip() != "")
    )
    df_reviews = df_reviews[has_text].copy()

    # 2. Agregar categorias de produto por pedido (um pedido pode ter múltiplos itens)
    df_items_prod = df_items.merge(
        df_products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )
    order_categories = (
        df_items_prod.groupby("order_id")["product_category_name"]
        .apply(lambda cats: ", ".join(sorted(set(cats.dropna()))))
        .reset_index()
    )

    # 3. Enriquecer com dados de entrega da tabela de pedidos
    orders_subset = df_orders[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].copy()

    # Flag de atraso de entrega
    delivered = pd.to_datetime(orders_subset["order_delivered_customer_date"])
    estimated = pd.to_datetime(orders_subset["order_estimated_delivery_date"])
    orders_subset["is_delayed"] = (delivered > estimated).fillna(False)

    # 4. Merges finais
    df_merged = df_reviews.merge(orders_subset, on="order_id", how="left")
    df_merged = df_merged.merge(order_categories, on="order_id", how="left")
    

    return df_merged