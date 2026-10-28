"""Download e preparação dos dados públicos da Olist (marketplace brasileiro).

Uso: python -m copiloto.data           # baixa os CSVs e gera data/reviews.csv

Rótulos REAIS (nada é anotado à mão):
- sentimento: nota do cliente 1-2 -> negativo, 3 -> neutro, 4-5 -> positivo;
- atrasado: data real de entrega > data prometida no momento da compra.

Fonte: "Brazilian E-Commerce Public Dataset by Olist" (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "reviews.csv"
MIRROR = "https://huggingface.co/datasets/miminmoons/olist-ecommerce-for-delivery-and-review-prediction/resolve/main/data"
FILES = [
    "olist_orders_dataset", "olist_order_reviews_dataset", "olist_order_items_dataset",
    "olist_products_dataset", "product_category_name_translation",
]

SENTIMENT = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}


def download(raw: Path = RAW) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw / f"{name}.csv"
        if not target.exists():
            print(f"baixando {name}...", file=sys.stderr)
            urllib.request.urlretrieve(f"{MIRROR}/{name}.csv", target)


def build(raw: Path = RAW) -> pd.DataFrame:
    dates = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    orders = pd.read_csv(raw / "olist_orders_dataset.csv", parse_dates=dates)
    reviews = pd.read_csv(raw / "olist_order_reviews_dataset.csv")
    items = pd.read_csv(raw / "olist_order_items_dataset.csv")
    products = pd.read_csv(raw / "olist_products_dataset.csv")

    # Um pedido pode ter itens de vários vendedores; ficamos com pedidos de vendedor único
    # para que a avaliação seja atribuível a um parceiro sem ambiguidade.
    sellers_per_order = items.groupby("order_id").seller_id.nunique()
    single = sellers_per_order[sellers_per_order == 1].index
    first_item = (items[items.order_id.isin(single)].sort_values("order_item_id")
                  .drop_duplicates("order_id").merge(products[["product_id", "product_category_name"]], on="product_id"))

    df = (reviews.dropna(subset=["review_comment_message"])
          .merge(orders[orders.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"]), on="order_id")
          .merge(first_item[["order_id", "seller_id", "product_category_name"]], on="order_id"))
    df = df.drop_duplicates("review_id")

    title = df.review_comment_title.fillna("").str.strip()
    body = df.review_comment_message.str.replace(r"\s+", " ", regex=True).str.strip()
    out = pd.DataFrame({
        "review_id": df.review_id,
        "order_id": df.order_id,
        "seller_id": df.seller_id,
        "categoria": df.product_category_name.fillna("sem_categoria"),
        "data_compra": df.order_purchase_timestamp.dt.date.astype(str),
        "nota": df.review_score.astype(int),
        "sentimento": df.review_score.map(SENTIMENT),
        "atrasado": (df.order_delivered_customer_date.dt.normalize() > df.order_estimated_delivery_date).astype(int),
        "dias_atraso": (df.order_delivered_customer_date.dt.normalize() - df.order_estimated_delivery_date).dt.days.clip(lower=0),
        "texto": (title + ". " + body).where(title != "", body),
    })
    return out[out.texto.str.len() >= 3].sort_values("data_compra").reset_index(drop=True)


def load(path: Path = PROCESSED) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} não existe. Rode: python -m copiloto.data")
    return pd.read_csv(path, dtype={"seller_id": str})


if __name__ == "__main__":
    download()
    data = build()
    data.to_csv(PROCESSED, index=False)
    print(f"{len(data)} avaliações com texto · {data.seller_id.nunique()} parceiros · "
          f"{data.atrasado.mean():.1%} atrasadas -> {PROCESSED}")
