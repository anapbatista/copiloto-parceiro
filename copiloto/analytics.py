"""Ferramentas analíticas do copiloto. Toda conta é feita aqui, em código testado;
o LLM (quando usado) só escolhe ferramentas e redige. Princípio: a IA interpreta,
o código calcula."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import ReviewClassifier
from .reasons import ACTIONS, LABELS, tag

MIN_PEER_REVIEWS = 20


def _pct(x: float) -> float:
    return round(100 * float(x), 1)


@dataclass
class Analytics:
    df: pd.DataFrame

    @classmethod
    def build(cls, reviews: pd.DataFrame, model: ReviewClassifier) -> Analytics:
        df = reviews.reset_index(drop=True).copy()
        df["seller_id"] = df["seller_id"].astype(str)
        df = pd.concat([df, model.predict(df.texto)], axis=1)
        df["motivos"] = df.texto.map(tag)
        df["mes"] = df.data_compra.str[:7]
        return cls(df)

    # ---------- utilidades ----------
    def _seller(self, seller_id: str) -> pd.DataFrame:
        sub = self.df[self.df.seller_id == seller_id]
        if sub.empty:
            raise KeyError(f"parceiro {seller_id} não encontrado")
        return sub

    def top_sellers(self, n: int = 10, min_reviews: int = 30) -> list[dict]:
        g = self.df.groupby("seller_id").agg(avaliacoes=("nota", "size"), nota_media=("nota", "mean"))
        g = g[g.avaliacoes >= min_reviews].sort_values("avaliacoes", ascending=False).head(n)
        return [{"seller_id": s, "avaliacoes": int(r.avaliacoes), "nota_media": round(r.nota_media, 2)} for s, r in g.iterrows()]

    # ---------- ferramentas ----------
    def perfil_parceiro(self, seller_id: str) -> dict:
        """Números gerais do parceiro, com tendência dos últimos 3 meses."""
        s = self._seller(seller_id)
        months = sorted(s.mes.unique())
        recent = s[s.mes.isin(months[-3:])]
        before = s[~s.mes.isin(months[-3:])]
        return {
            "seller_id": seller_id,
            "categoria_principal": s.categoria.mode().iat[0],
            "periodo": f"{s.data_compra.min()} a {s.data_compra.max()}",
            "avaliacoes_com_texto": int(len(s)),
            "nota_media": round(float(s.nota.mean()), 2),
            "pct_negativas": _pct((s.nota <= 2).mean()),
            "pct_pedidos_atrasados": _pct(s.atrasado.mean()),
            "pct_negativas_ultimos_3_meses": _pct((recent.nota <= 2).mean()) if len(recent) else None,
            "pct_negativas_antes": _pct((before.nota <= 2).mean()) if len(before) else None,
            "alertas_nota_alta_texto_negativo": int(((s.nota >= 4) & (s.sentimento_pred == "negativo")).sum()),
        }

    def motivos_reclamacao(self, seller_id: str) -> dict:
        """Distribuição dos motivos entre avaliações com nota <= 3, com ação recomendada."""
        s = self._seller(seller_id)
        complaints = s[s.nota <= 3]
        counts = complaints.motivos.explode().dropna().value_counts()
        total = len(complaints)
        return {
            "seller_id": seller_id,
            "avaliacoes_nota_ate_3": int(total),
            "pct_com_motivo_identificado": _pct((complaints.motivos.str.len() > 0).mean()) if total else 0.0,
            "motivos": [
                {"motivo": m, "descricao": LABELS[m], "ocorrencias": int(c), "pct_das_reclamacoes": _pct(c / total),
                 "acao_recomendada": ACTIONS[m]}
                for m, c in counts.items()
            ],
        }

    def exemplos(self, seller_id: str, motivo: str, k: int = 3) -> dict:
        """Trechos literais (mais recentes) de reclamações com o motivo informado."""
        s = self._seller(seller_id)
        hits = s[(s.nota <= 3) & s.motivos.map(lambda m: motivo in m)].sort_values("data_compra", ascending=False)
        return {
            "seller_id": seller_id, "motivo": motivo,
            "exemplos": [{"data": r.data_compra, "nota": int(r.nota), "texto": r.texto[:220]} for r in hits.head(k).itertuples()],
        }

    def comparar_com_categoria(self, seller_id: str) -> dict:
        """Posição do parceiro frente a parceiros da mesma categoria (percentil: 100 = pior)."""
        s = self._seller(seller_id)
        cat = s.categoria.mode().iat[0]
        peers = self.df[self.df.categoria == cat].groupby("seller_id").agg(
            n=("nota", "size"), neg=("nota", lambda x: (x <= 2).mean()), atraso=("atrasado", "mean"))
        peers = peers[peers.n >= MIN_PEER_REVIEWS]
        me_neg, me_late = (s.nota <= 2).mean(), s.atrasado.mean()
        return {
            "seller_id": seller_id, "categoria": cat, "parceiros_comparados": int(len(peers)),
            "pct_negativas_parceiro": _pct(me_neg), "pct_negativas_mediana_categoria": _pct(peers.neg.median()) if len(peers) else None,
            "percentil_negativas": _pct((peers.neg < me_neg).mean()) if len(peers) else None,
            "pct_atraso_parceiro": _pct(me_late), "pct_atraso_mediana_categoria": _pct(peers.atraso.median()) if len(peers) else None,
            "percentil_atraso": _pct((peers.atraso < me_late).mean()) if len(peers) else None,
        }

    def call(self, name: str, **kwargs) -> dict:
        tools = {"perfil_parceiro": self.perfil_parceiro, "motivos_reclamacao": self.motivos_reclamacao,
                 "exemplos": self.exemplos, "comparar_com_categoria": self.comparar_com_categoria}
        if name not in tools:
            raise KeyError(f"ferramenta desconhecida: {name}")
        return tools[name](**kwargs)


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "perfil_parceiro", "description": "Números gerais do parceiro: nota média, % negativas, % atrasos, tendência.",
        "parameters": {"type": "object", "properties": {"seller_id": {"type": "string"}}, "required": ["seller_id"]}}},
    {"type": "function", "function": {
        "name": "motivos_reclamacao", "description": "Motivos das reclamações (nota <= 3) com % e ação recomendada.",
        "parameters": {"type": "object", "properties": {"seller_id": {"type": "string"}}, "required": ["seller_id"]}}},
    {"type": "function", "function": {
        "name": "exemplos", "description": "Trechos literais de reclamações de um motivo.",
        "parameters": {"type": "object", "properties": {
            "seller_id": {"type": "string"}, "motivo": {"type": "string", "enum": list(LABELS)}, "k": {"type": "integer"}},
            "required": ["seller_id", "motivo"]}}},
    {"type": "function", "function": {
        "name": "comparar_com_categoria", "description": "Compara o parceiro com a mediana da categoria.",
        "parameters": {"type": "object", "properties": {"seller_id": {"type": "string"}}, "required": ["seller_id"]}}},
]
