"""Mensagens de saída. Regra de privacidade: o grupo vê O QUE cada um vai comer e de
onde, mas nunca valores individuais nem orçamentos. Valores vão só no privado."""

from __future__ import annotations

from .models import Participant, Plan, Restaurant


def brl(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def group_message(plan: Plan, people: dict[str, Participant], restaurants: dict[str, Restaurant]) -> str:
    if not plan.assignments:
        return "Não consegui montar um pedido que respeite os orçamentos de todo mundo. Que tal ajustar os pedidos?"
    lines = [f"Pedido fechado em {len(plan.restaurant_ids)} restaurante(s):"]
    for rid in plan.restaurant_ids:
        r = restaurants[rid]
        lines.append(f"\n*{r.name}* (chega em ~{r.eta_min} min)")
        for a in plan.assignments:
            if a.restaurant_id == rid:
                lines.append(f"- {people[a.participant_id].name}: {a.item.name}")
    if plan.unserved:
        names = ", ".join(people[p].name for p in plan.unserved)
        lines.append(f"\nNão encontrei opção dentro do combinado para: {names}. Te chamo no privado.")
    lines.append("\nOs valores de cada um foram enviados no privado.")
    return "\n".join(lines)


def private_message(participant: Participant, plan: Plan, restaurants: dict[str, Restaurant]) -> str:
    a = plan.assignment_for(participant.id)
    if a is None:
        return (f"Oi, {participant.name}! Com o limite de {brl(participant.budget)} não achei nada parecido com o "
                "que você pediu nos restaurantes escolhidos. Quer aumentar o limite ou trocar o pedido?")
    sobra = participant.budget - a.total
    return (f"Oi, {participant.name}! Seu pedido: {a.item.name} ({restaurants[a.restaurant_id].name}).\n"
            f"Item: {brl(a.item.price)} + sua parte da entrega: {brl(a.fee_share)} = *{brl(a.total)}*.\n"
            f"Dentro do seu limite de {brl(participant.budget)} (sobram {brl(sobra)}).")
