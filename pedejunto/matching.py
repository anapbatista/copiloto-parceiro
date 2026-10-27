"""Casamento intenção -> item do cardápio.

Score em [0, 1] combinando:
- cobertura exata dos termos da pessoa (com stemming e sinônimos);
- meio ponto quando o termo casa só com a culinária do restaurante;
- similaridade fuzzy por trigramas (tolera erro de digitação: "hamburgue", "piza");
Restrições alimentares são filtro duro (score 0). Ingredientes excluídos penalizam.
"""

from __future__ import annotations

from .models import Intent, MenuItem, Restaurant
from .text import SYNONYMS, normalize, stem, tokenize, trigrams


def _expand(term: str) -> set[str]:
    out = {stem(term)}
    for syn in SYNONYMS.get(term, []):
        out.add(stem(syn))
    return out


def _tokens(text: str) -> set[str]:
    return {stem(t) for t in tokenize(text)}


def _fuzzy(term: str, tokens: set[str]) -> float:
    tg = trigrams(stem(term))
    best = 0.0
    for tok in tokens:
        other = trigrams(tok)
        best = max(best, len(tg & other) / len(tg | other))
    return best


def match_score(intent: Intent, item: MenuItem, restaurant: Restaurant | None = None) -> float:
    if not intent.restrictions.issubset(item.tags):
        return 0.0
    tokens = _tokens(f"{item.name} {item.description}")
    cuisine = _tokens(restaurant.cuisine) if restaurant is not None else set()
    if not intent.terms:
        # Pessoa só disse "qualquer coisa vegana": qualquer item compatível serve.
        return 0.5 if intent.restrictions else 0.3

    per_term = []
    for term in intent.terms:
        expanded = _expand(term)
        if expanded & tokens:
            per_term.append(1.0)
        elif expanded & cuisine:
            per_term.append(0.5)  # "japonês" casa com o restaurante, mas não diz qual prato
        else:
            fz = _fuzzy(term, tokens)
            per_term.append(fz if fz >= 0.45 else 0.0)
    score = sum(per_term) / len(per_term)

    item_text = normalize(f"{item.name} {item.description}")
    for excluded in intent.excluded_terms:
        if excluded and excluded in item_text:
            score *= 0.7  # dá pra pedir "sem", mas um item que já não tem é melhor
    return round(score, 4)
