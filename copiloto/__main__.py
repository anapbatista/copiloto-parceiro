"""CLI do Copiloto do Parceiro.

python -m copiloto parceiros            # lista parceiros com mais avaliações
python -m copiloto relatorio <seller>   # plano de ação (usa LLM se COPILOTO_LLM_MODEL estiver definido)
"""

from __future__ import annotations

import argparse
import os

from .agent import Copilot
from .analytics import Analytics
from .data import load
from .model import ReviewClassifier


def build_client():
    model = os.getenv("COPILOTO_LLM_MODEL")
    if not model:
        return None, None
    from openai import OpenAI  # opcional: pip install .[llm]

    return OpenAI(base_url=os.getenv("COPILOTO_LLM_BASE_URL")), model


def main() -> None:
    ap = argparse.ArgumentParser(prog="copiloto")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parceiros")
    p.add_argument("--n", type=int, default=10)
    r = sub.add_parser("relatorio")
    r.add_argument("seller_id")
    args = ap.parse_args()

    an = Analytics.build(load(), ReviewClassifier.load())
    if args.cmd == "parceiros":
        for row in an.top_sellers(args.n):
            print(f"{row['seller_id']}  avaliações={row['avaliacoes']:4d}  nota={row['nota_media']}")
        return
    client, model = build_client()
    result = Copilot(an, client, model).run(args.seller_id)
    print(result.report)
    f = result.faithfulness
    print(f"\n---\nmodo: {result.mode} · ferramentas: {', '.join(result.tool_calls)} · "
          f"fidelidade: {'OK' if f.ok else 'FALHOU'} ({f.numbers_checked} números, {f.quotes_checked} citações verificados)")


if __name__ == "__main__":
    main()
