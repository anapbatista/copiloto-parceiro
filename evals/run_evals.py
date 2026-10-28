"""Avaliação do Copiloto do Parceiro.

1. Classificadores (divisão temporal, teste abr-ago/2018): sentimento e risco de atraso.
2. Regras de motivo: concordância da regra "atraso / não recebido" com o atraso REAL do pedido.
3. Fidelidade do relatório: em N parceiros, todo número e citação conferem com as ferramentas?
   E o verificador pega alucinações? (injetamos números e citações falsas e medimos a detecção)
4. Latência para gerar um relatório.

Uso: python evals/run_evals.py [--sellers 50] [--markdown]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import precision_score, recall_score  # noqa: E402

from copiloto.agent import deterministic_report  # noqa: E402
from copiloto.analytics import Analytics  # noqa: E402
from copiloto.data import load  # noqa: E402
from copiloto.faithfulness import check  # noqa: E402
from copiloto.model import temporal_split, train  # noqa: E402
from copiloto.reasons import tag  # noqa: E402


def eval_rules(test):
    rule = test.texto.map(lambda t: bool({"atraso", "nao_recebido"} & set(tag(t)))).astype(int)
    return {"precisao": round(precision_score(test.atrasado, rule), 4), "recall": round(recall_score(test.atrasado, rule), 4)}


def eval_faithfulness(an: Analytics, n: int, seed: int = 0):
    rng = random.Random(seed)
    sellers = [s["seller_id"] for s in an.top_sellers(n, min_reviews=30)]
    ok, numbers, quotes, caught_num, caught_quote, latencies = 0, 0, 0, 0, 0, []
    for sid in sellers:
        t0 = time.perf_counter()
        report, evidence, _ = deterministic_report(an, sid)
        latencies.append((time.perf_counter() - t0) * 1000)
        res = check(report, evidence)
        ok += res.ok
        numbers += res.numbers_checked
        quotes += res.quotes_checked
        # Alucinações sintéticas: um percentual inventado e uma citação inventada.
        fake_pct = f"{rng.randint(11, 99)},{rng.randint(1, 9)}%"
        caught_num += not check(report + f"\n- Além disso, {fake_pct} dos clientes pediram reembolso.", evidence).ok
        caught_quote += not check(report + "\n> “Produto excelente, chegou antes do prazo e muito bem embalado!!”", evidence).ok
    return {
        "parceiros": len(sellers), "relatorios_fieis": ok, "numeros_verificados": numbers, "citacoes_verificadas": quotes,
        "deteccao_numero_inventado": round(caught_num / len(sellers), 4),
        "deteccao_citacao_inventada": round(caught_quote / len(sellers), 4),
        "latencia_p50_ms": round(statistics.median(latencies), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sellers", type=int, default=50)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    df = load()
    model, clf = train(df)
    _, _, test = temporal_split(df)
    rules = eval_rules(test)
    faith = eval_faithfulness(Analytics.build(df, model), args.sellers)

    if not args.markdown:
        print(json.dumps({"classificadores": clf, "regra_atraso": rules, "fidelidade": faith}, indent=2, ensure_ascii=False))
        return
    print("| Métrica | Resultado |\n|---|---|")
    print(f"| Sentimento: macro-F1 / acurácia ({clf['n_teste']} avaliações de teste) | "
          f"{clf['sentimento_macro_f1']:.3f} / {clf['sentimento_acuracia']:.3f} |")
    print(f"| Sentimento: F1 da classe negativa | {clf['sentimento_f1_negativo']:.3f} |")
    print(f"| Risco de atraso pelo texto: ROC-AUC / PR-AUC (taxa base {clf['atraso_taxa_base']:.1%}) | "
          f"{clf['atraso_roc_auc']:.3f} / {clf['atraso_pr_auc']:.3f} |")
    print(f"| Regra \"atraso/não recebido\" vs. atraso real: precisão / recall | {rules['precisao']:.3f} / {rules['recall']:.3f} |")
    print(f"| Relatórios 100% fiéis aos dados | {faith['relatorios_fieis']}/{faith['parceiros']} "
          f"({faith['numeros_verificados']} números, {faith['citacoes_verificadas']} citações) |")
    print(f"| Detecção de número inventado / citação inventada | "
          f"{faith['deteccao_numero_inventado']:.0%} / {faith['deteccao_citacao_inventada']:.0%} |")
    print(f"| Latência para gerar um relatório (p50) | {faith['latencia_p50_ms']:.0f} ms |")


if __name__ == "__main__":
    main()
