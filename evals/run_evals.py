"""Avaliação offline do PedeJunto.

1. Entendimento de intenção (30 frases rotuladas, linguagem de WhatsApp):
   - acurácia de restrições alimentares (match exato do conjunto);
   - top-1 retrieval: o melhor item do catálogo é um dos esperados?
2. Otimizador vs. força bruta (cenários aleatórios com seed fixa):
   - % de cenários em que o plano é ótimo (mesmo nº de servidos e mesma utilidade);
   - violações de orçamento e de limite de restaurantes (devem ser 0);
3. Latência em grupos maiores (catálogo completo).

Uso: python evals/run_evals.py [--scenarios 300] [--markdown]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pedejunto.catalog import JsonCatalog  # noqa: E402
from pedejunto.intent import RuleBasedIntentParser  # noqa: E402
from pedejunto.matching import match_score  # noqa: E402
from pedejunto.models import Participant  # noqa: E402
from pedejunto.optimizer import OptimizerConfig, _candidates, plan_group_order  # noqa: E402

CFG = OptimizerConfig()


def eval_intents(restaurants):
    data = json.loads((ROOT / "evals" / "intents.json").read_text(encoding="utf-8"))
    parser = RuleBasedIntentParser()
    restr_ok = top1_ok = 0
    failures = []
    for i, ex in enumerate(data):
        intent = parser.parse(f"p{i}", ex["text"])
        restr_ok += intent.restrictions == set(ex["restrictions"])
        best = max(((match_score(intent, it, r), it) for r in restaurants for it in r.items), key=lambda x: x[0])
        exp = ex["expected_item_any"]
        hit = ("vegetariano" in best[1].tags) if exp == ["*veg"] else best[1].id in exp
        top1_ok += hit
        if not hit:
            failures.append((ex["text"], best[1].name))
    n = len(data)
    return {"n": n, "restriction_acc": restr_ok / n, "top1_acc": top1_ok / n, "failures": failures}


def brute_force(people, intents, restaurants):
    rmap = {r.id: r for r in restaurants}
    cand = _candidates(intents, restaurants, CFG)
    options = [[None] + [(rid, it, s) for rid, lst in cand[p.id].items() for it, s in lst] for p in people]
    best = (-1, float("-inf"))
    for combo in product(*options):
        used = {o[0] for o in combo if o}
        if len(used) > CFG.max_restaurants:
            continue
        counts = {r: sum(1 for o in combo if o and o[0] == r) for r in used}
        ok, util, subtotal = True, 0.0, {r: 0.0 for r in used}
        for p, o in zip(people, combo, strict=True):
            if o is None:
                continue
            rid, it, s = o
            total = it.price + round(rmap[rid].delivery_fee / counts[rid], 2)
            if total > p.budget + 1e-9:
                ok = False
                break
            util += s - CFG.price_weight * total / p.budget
            subtotal[rid] += it.price
        if not ok or any(subtotal[r] + 1e-9 < rmap[r].min_order for r in used):
            continue
        served = sum(o is not None for o in combo)
        if (served, round(util, 9)) > (best[0], round(best[1], 9)):
            best = (served, util)
    return best


def random_scenario(rng, texts, restaurants, n_people, n_rest):
    rs = rng.sample(restaurants, n_rest)
    people = [Participant(f"p{i}", f"P{i}", float(rng.randint(18, 55))) for i in range(n_people)]
    parser = RuleBasedIntentParser()
    intents = {p.id: parser.parse(p.id, rng.choice(texts)) for p in people}
    return people, intents, rs


def eval_optimizer(restaurants, n_scenarios, seed=42):
    rng = random.Random(seed)
    texts = [ex["text"] for ex in json.loads((ROOT / "evals" / "intents.json").read_text(encoding="utf-8"))]
    optimal = budget_viol = rest_viol = 0
    gaps = []
    for _ in range(n_scenarios):
        people, intents, rs = random_scenario(rng, texts, restaurants, rng.randint(2, 5), rng.randint(3, 6))
        plan = plan_group_order(people, intents, rs, CFG)
        bmap = {p.id: p.budget for p in people}
        budget_viol += any(a.total > bmap[a.participant_id] + 1e-9 for a in plan.assignments)
        rest_viol += len({a.restaurant_id for a in plan.assignments}) > CFG.max_restaurants
        bf_served, bf_util = brute_force(people, intents, rs)
        if plan.served == bf_served and abs(plan.utility - bf_util) < 1e-6:
            optimal += 1
        elif plan.served == bf_served:
            gaps.append(bf_util - plan.utility)
        else:
            gaps.append(float("nan"))
    return {
        "n": n_scenarios,
        "optimal_rate": optimal / n_scenarios,
        "budget_violations": budget_viol,
        "restaurant_violations": rest_viol,
        "non_optimal": len(gaps),
    }


def eval_latency(restaurants, sizes=(6, 10, 15), reps=20, seed=7):
    rng = random.Random(seed)
    texts = [ex["text"] for ex in json.loads((ROOT / "evals" / "intents.json").read_text(encoding="utf-8"))]
    out = {}
    for n in sizes:
        times = []
        for _ in range(reps):
            people, intents, rs = random_scenario(rng, texts, restaurants, n, len(restaurants))
            t0 = time.perf_counter()
            plan_group_order(people, intents, rs, CFG)
            times.append((time.perf_counter() - t0) * 1000)
        out[n] = (statistics.median(times), sorted(times)[int(0.95 * (reps - 1))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    restaurants = JsonCatalog().list_restaurants()

    it = eval_intents(restaurants)
    op = eval_optimizer(restaurants, args.scenarios)
    lat = eval_latency(restaurants)

    if args.markdown:
        print("| Métrica | Resultado |\n|---|---|")
        print(f"| Restrições alimentares (acurácia, {it['n']} frases) | {it['restriction_acc']:.0%} |")
        print(f"| Item certo no top-1 ({it['n']} frases) | {it['top1_acc']:.0%} |")
        print(f"| Planos ótimos vs. força bruta ({op['n']} cenários) | {op['optimal_rate']:.1%} |")
        print(f"| Violações de orçamento | {op['budget_violations']} |")
        print(f"| Violações do limite de 2 restaurantes | {op['restaurant_violations']} |")
        for n, (p50, p95) in lat.items():
            print(f"| Latência do otimizador, {n} pessoas, 8 restaurantes (p50 / p95) | {p50:.0f} ms / {p95:.0f} ms |")
    else:
        print(json.dumps({"intents": it, "optimizer": op, "latency_ms": lat}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
