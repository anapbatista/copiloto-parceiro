"""Sessão de pedido em grupo: o "cérebro" que o bot de WhatsApp chama.

Fluxo:
1. Cada pessoa escreve no GRUPO o que quer comer (texto livre).
2. Cada pessoa manda no PRIVADO o seu limite ("meu limite é 40").
3. Alguém digita "fechar" -> o agente interpreta, otimiza e responde:
   resumo no grupo + detalhamento de valores no privado de cada um.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import CatalogProvider, JsonCatalog
from .intent import IntentParser, RuleBasedIntentParser
from .messages import group_message, private_message
from .models import Intent, Participant, Plan
from .optimizer import OptimizerConfig, plan_group_order

_MONEY = re.compile(r"(\d+(?:[.,]\d{1,2})?)")


def parse_budget(text: str) -> float | None:
    """Extrai valor de mensagens como 'meu limite é R$ 42,50' ou 'até 40 conto'."""
    m = _MONEY.search(text.replace("R$", " "))
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    return value if value > 0 else None


@dataclass
class OrderResult:
    plan: Plan
    group_text: str
    private_texts: dict[str, str]


@dataclass
class GroupOrderSession:
    catalog: CatalogProvider = field(default_factory=JsonCatalog)
    parser: IntentParser = field(default_factory=RuleBasedIntentParser)
    config: OptimizerConfig = field(default_factory=OptimizerConfig)
    names: dict[str, str] = field(default_factory=dict)
    wishes: dict[str, list[str]] = field(default_factory=dict)
    budgets: dict[str, float] = field(default_factory=dict)

    def on_group_message(self, participant_id: str, name: str, text: str) -> None:
        self.names[participant_id] = name
        self.wishes.setdefault(participant_id, []).append(text)

    def on_private_message(self, participant_id: str, text: str) -> str:
        value = parse_budget(text)
        if value is None:
            return "Não entendi o valor. Me manda só o número, tipo: 45"
        self.budgets[participant_id] = value
        return f"Anotado! Seu limite fica só comigo: R$ {value:.2f}".replace(".", ",")

    def pending_budgets(self) -> list[str]:
        return [self.names[p] for p in self.wishes if p not in self.budgets]

    def close(self) -> OrderResult:
        people = {
            pid: Participant(pid, self.names[pid], self.budgets[pid]) for pid in self.wishes if pid in self.budgets
        }
        intents: dict[str, Intent] = {
            pid: self.parser.parse(pid, " ".join(self.wishes[pid])) for pid in people
        }
        restaurants = self.catalog.list_restaurants()
        plan = plan_group_order(list(people.values()), intents, restaurants, self.config)
        rmap = {r.id: r for r in restaurants}
        return OrderResult(
            plan=plan,
            group_text=group_message(plan, people, rmap),
            private_texts={pid: private_message(p, plan, rmap) for pid, p in people.items()},
        )
