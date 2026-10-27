"""Estruturas de dados do domínio: catálogo, participantes, intenções e plano."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MenuItem:
    id: str
    name: str
    price: float
    tags: frozenset[str] = frozenset()
    description: str = ""


@dataclass(frozen=True)
class Restaurant:
    id: str
    name: str
    cuisine: str
    delivery_fee: float
    min_order: float
    eta_min: int
    items: tuple[MenuItem, ...]


@dataclass(frozen=True)
class Participant:
    """Pessoa do grupo. O orçamento é privado: nunca aparece em mensagens do grupo."""

    id: str
    name: str
    budget: float


@dataclass
class Intent:
    """O que a pessoa quer comer, extraído de texto livre."""

    participant_id: str
    raw_text: str
    terms: list[str] = field(default_factory=list)
    restrictions: set[str] = field(default_factory=set)  # vegetariano, vegano, sem_lactose, sem_gluten
    excluded_terms: list[str] = field(default_factory=list)  # "sem cebola" -> ["cebola"]


@dataclass(frozen=True)
class Assignment:
    participant_id: str
    restaurant_id: str
    item: MenuItem
    fee_share: float
    match_score: float

    @property
    def total(self) -> float:
        return round(self.item.price + self.fee_share, 2)


@dataclass
class Plan:
    restaurant_ids: list[str]
    assignments: list[Assignment]
    unserved: list[str]
    utility: float

    def assignment_for(self, participant_id: str) -> Assignment | None:
        return next((a for a in self.assignments if a.participant_id == participant_id), None)

    @property
    def served(self) -> int:
        return len(self.assignments)
