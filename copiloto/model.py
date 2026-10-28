"""Classificadores de avaliações.

Duas tarefas com rótulos reais:
- sentimento (negativo / neutro / positivo), a partir da nota;
- risco de atraso: o texto indica que o pedido chegou depois do prometido?

Baseline: TF-IDF (palavras + n-gramas de caracteres, robusto a erro de digitação)
+ regressão logística balanceada. O fine-tuning do BERTimbau fica em train/.

Divisão TEMPORAL: treino até fev/2018, validação mar/2018, teste abr-ago/2018.
Dividir por tempo (e não aleatoriamente) mede o que importa em produção: o modelo
treinado no passado funciona nas avaliações que chegam depois?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .text import normalize

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "baseline.joblib"
VAL_START, TEST_START = "2018-03-01", "2018-04-01"


def temporal_split(df: pd.DataFrame):
    train = df[df.data_compra < VAL_START]
    val = df[(df.data_compra >= VAL_START) & (df.data_compra < TEST_START)]
    test = df[df.data_compra >= TEST_START]
    return train, val, test


def _pipeline(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=80000)),
    ])
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


@dataclass
class ReviewClassifier:
    sentiment: Pipeline
    delay: Pipeline
    delay_threshold: float

    def predict(self, texts: list[str]) -> pd.DataFrame:
        texts = list(texts)
        proba = self.delay.predict_proba(texts)[:, 1]
        return pd.DataFrame({
            "sentimento_pred": self.sentiment.predict(texts),
            "risco_atraso": proba.round(4),
            "atraso_pred": (proba >= self.delay_threshold).astype(int),
        })

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> ReviewClassifier:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} não existe. Rode: python -m copiloto.model")
        return joblib.load(path)


def _best_threshold(y, proba) -> float:
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def train(df: pd.DataFrame, C_sent: float = 1.0, C_delay: float = 0.5) -> tuple[ReviewClassifier, dict]:
    tr, va, te = temporal_split(df)
    sent = _pipeline(C_sent).fit(tr.texto, tr.sentimento)
    delay = _pipeline(C_delay).fit(tr.texto, tr.atrasado)
    threshold = _best_threshold(va.atrasado, delay.predict_proba(va.texto)[:, 1])  # escolhido na validação
    model = ReviewClassifier(sent, delay, threshold)
    return model, evaluate(model, te)


def evaluate(model: ReviewClassifier, test: pd.DataFrame) -> dict:
    pred = model.predict(test.texto)
    return {
        "n_teste": int(len(test)),
        "sentimento_macro_f1": round(f1_score(test.sentimento, pred.sentimento_pred, average="macro"), 4),
        "sentimento_acuracia": round(accuracy_score(test.sentimento, pred.sentimento_pred), 4),
        "sentimento_f1_negativo": round(f1_score(test.sentimento, pred.sentimento_pred, labels=["negativo"], average="macro"), 4),
        "atraso_roc_auc": round(roc_auc_score(test.atrasado, pred.risco_atraso), 4),
        "atraso_pr_auc": round(average_precision_score(test.atrasado, pred.risco_atraso), 4),
        "atraso_f1": round(f1_score(test.atrasado, pred.atraso_pred), 4),
        "atraso_taxa_base": round(float(test.atrasado.mean()), 4),
    }


if __name__ == "__main__":
    import json

    from .data import load

    model, metrics = train(load())
    model.save()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
