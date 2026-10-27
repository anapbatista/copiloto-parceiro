"""Otimizador do pedido em grupo.

Problema: cada pessoa escolhe no máximo 1 item; o grupo usa no máximo K restaurantes
(K=2 por padrão); a taxa de entrega de cada restaurante é dividida igualmente entre
quem pediu nele; o custo de cada pessoa (item + sua parte da taxa) não pode passar do
SEU orçamento privado; o subtotal de cada restaurante precisa atingir o pedido mínimo.

Objetivo lexicográfico: (1) servir o máximo de pessoas; (2) maximizar a utilidade total,
onde utilidade = aderência ao desejo - peso * custo relativo ao orçamento.

O ponto difícil é que a parte da taxa depende de QUANTAS pessoas ficam em cada
restaurante (não-linear). A solução fixa os contadores (kA, kB): com eles fixos, a taxa
por pessoa vira constante e o problema se decompõe.

- Caminho rápido (todos servidos): para (A, B, kA) fixos, escolher quais kA pessoas vão
  para A é exato por ordenação da diferença de utilidade uA - uB (argumento de troca).
- Fallback (nem todos cabem no orçamento): programação dinâmica sobre pessoas com estado
  (a, b) = quantos já foram para A e B, permitindo deixar alguém sem pedido.
- Pedido mínimo: se a escolha gulosa viola o mínimo de algum restaurante, o caminho
  rápido faz busca exata entre os C(n, kA) subconjuntos (limitada a MAX_SUBSETS).

Complexidade com R restaurantes, n pessoas e m itens por cardápio:
caminho rápido O(R² · n · (n·m + n log n)); fallback O(R² · n³ · m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from .matching import match_score
from .models import Assignment, Intent, MenuItem, Participant, Plan, Restaurant

NEG = -math.inf
MAX_SUBSETS = 5000  # grupos de até ~14 pessoas ficam exatos mesmo com pedido mínimo


@dataclass(frozen=True)
class OptimizerConfig:
    max_restaurants: int = 2
    min_match: float = 0.34
    price_weight: float = 0.15


def _candidates(intents: dict[str, Intent], restaurants: list[Restaurant], cfg: OptimizerConfig):
    """cand[pid][rid] = [(item, score), ...] apenas com itens aderentes ao desejo."""
    cand: dict[str, dict[str, list[tuple[MenuItem, float]]]] = {}
    for pid, intent in intents.items():
        cand[pid] = {}
        for r in restaurants:
            opts = [(it, s) for it in r.items if (s := match_score(intent, it, r)) >= cfg.min_match]
            if opts:
                cand[pid][r.id] = opts
    return cand


def _best(options, fee_share: float, budget: float, cfg: OptimizerConfig):
    """Melhor item (utilidade, item, score) que cabe no orçamento com a taxa dada."""
    best = (NEG, None, 0.0)
    fee_share = round(fee_share, 2)  # o que a pessoa paga de fato
    for item, score in options or ():
        total = item.price + fee_share
        if total <= budget + 1e-9:
            u = score - cfg.price_weight * total / budget
            if u > best[0] or (u == best[0] and best[1] is not None and item.price < best[1].price):
                best = (u, item, score)
    return best


def _min_order_ok(assign: list[Assignment], rmap: dict[str, Restaurant]) -> bool:
    subtotal: dict[str, float] = {}
    for a in assign:
        subtotal[a.restaurant_id] = subtotal.get(a.restaurant_id, 0.0) + a.item.price
    return all(v + 1e-9 >= rmap[rid].min_order for rid, v in subtotal.items())


def _better(plan: Plan | None, best: Plan | None) -> bool:
    if plan is None:
        return False
    if best is None:
        return True
    return (plan.served, round(plan.utility, 9)) > (best.served, round(best.utility, 9))


def _all_served(people, combo, cand, rmap, cfg) -> Plan | None:
    n = len(people)
    best_plan = None
    if len(combo) == 1:
        (rid,) = combo
        share = rmap[rid].delivery_fee / n
        picks = [_best(cand[p.id].get(rid), share, p.budget, cfg) for p in people]
        if any(u == NEG for u, _, _ in picks):
            return None
        assign = [Assignment(p.id, rid, it, round(share, 2), s) for p, (_, it, s) in zip(people, picks, strict=True)]
        if _min_order_ok(assign, rmap):
            return Plan([rid], assign, [], sum(u for u, _, _ in picks))
        return None

    ra, rb = combo
    for ka in range(1, n):
        sa, sb = rmap[ra].delivery_fee / ka, rmap[rb].delivery_fee / (n - ka)
        pa = [_best(cand[p.id].get(ra), sa, p.budget, cfg) for p in people]
        pb = [_best(cand[p.id].get(rb), sb, p.budget, cfg) for p in people]

        def key(i, pa=pa, pb=pb):
            ua, ub = pa[i][0], pb[i][0]
            if ua == NEG:
                return NEG
            if ub == NEG:
                return math.inf
            return ua - ub

        def build(to_a, pa=pa, pb=pb, sa=sa, sb=sb):
            total, assign = 0.0, []
            for i, p in enumerate(people):
                u, it, s = pa[i] if i in to_a else pb[i]
                if u == NEG:
                    return None
                rid, share = (ra, sa) if i in to_a else (rb, sb)
                assign.append(Assignment(p.id, rid, it, round(share, 2), s))
                total += u
            return Plan([ra, rb], assign, [], total) if _min_order_ok(assign, rmap) else None

        order = sorted(range(n), key=key, reverse=True)
        plan = build(set(order[:ka]))
        if plan is None and math.comb(n, ka) <= MAX_SUBSETS:
            # O pedido mínimo quebrou a ordenação gulosa: busca exata entre os subconjuntos.
            for subset in combinations(range(n), ka):
                cand_plan = build(set(subset))
                if _better(cand_plan, plan):
                    plan = cand_plan
        if _better(plan, best_plan):
            best_plan = plan
    return best_plan


def _partial(people, combo, cand, rmap, cfg) -> Plan | None:
    """DP que permite deixar pessoas sem pedido quando o orçamento não fecha."""
    n = len(people)
    ra = combo[0]
    rb = combo[1] if len(combo) > 1 else None
    best_plan = None
    for ka in range(1, n + 1):
        kb_range = range(1, n - ka + 1) if rb else range(0, 1)
        for kb in kb_range:
            sa = rmap[ra].delivery_fee / ka
            sb = rmap[rb].delivery_fee / kb if rb else 0.0
            pa = [_best(cand[p.id].get(ra), sa, p.budget, cfg) for p in people]
            pb = [_best(cand[p.id].get(rb), sb, p.budget, cfg) if rb else (NEG, None, 0) for p in people]
            # dp[(a, b)] = (utilidade, escolhas)
            dp = {(0, 0): (0.0, ())}
            for i in range(n):
                nxt = {}
                for (a, b), (u, ch) in dp.items():
                    moves = [((a, b), u, ch + (None,))]
                    if a < ka and pa[i][0] != NEG:
                        moves.append(((a + 1, b), u + pa[i][0], ch + ("A",)))
                    if b < kb and pb[i][0] != NEG:
                        moves.append(((a, b + 1), u + pb[i][0], ch + ("B",)))
                    for st, uu, cc in moves:
                        if st not in nxt or uu > nxt[st][0]:
                            nxt[st] = (uu, cc)
                dp = nxt
            if (ka, kb) not in dp:
                continue
            u, choices = dp[(ka, kb)]
            assign, unserved = [], []
            for i, (p, c) in enumerate(zip(people, choices, strict=True)):
                if c is None:
                    unserved.append(p.id)
                elif c == "A":
                    assign.append(Assignment(p.id, ra, pa[i][1], round(sa, 2), pa[i][2]))
                else:
                    assign.append(Assignment(p.id, rb, pb[i][1], round(sb, 2), pb[i][2]))
            if not _min_order_ok(assign, rmap):
                continue
            plan = Plan([r for r in combo if any(a.restaurant_id == r for a in assign)], assign, unserved, u)
            if _better(plan, best_plan):
                best_plan = plan
    return best_plan


def plan_group_order(
    participants: list[Participant],
    intents: dict[str, Intent],
    restaurants: list[Restaurant],
    config: OptimizerConfig | None = None,
) -> Plan:
    cfg = config or OptimizerConfig()
    people = [p for p in participants if p.id in intents]
    if not people:
        return Plan([], [], [], 0.0)
    rmap = {r.id: r for r in restaurants}
    cand = _candidates(intents, restaurants, cfg)
    useful = [r.id for r in restaurants if any(r.id in cand[p.id] for p in people)]

    combos = [(r,) for r in useful]
    if cfg.max_restaurants >= 2:
        combos += list(combinations(useful, 2))

    best = None
    for combo in combos:
        plan = _all_served(people, combo, cand, rmap, cfg)
        if _better(plan, best):
            best = plan
    if best is None:
        for combo in combos:
            plan = _partial(people, combo, cand, rmap, cfg)
            if _better(plan, best):
                best = plan
    if best is None:
        return Plan([], [], [p.id for p in people], 0.0)
    # Restaurante sem ninguém não entra no plano.
    best.restaurant_ids = [r for r in best.restaurant_ids if any(a.restaurant_id == r for a in best.assignments)]
    return best
