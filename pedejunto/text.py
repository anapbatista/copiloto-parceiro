"""Normalização de texto em português (sem dependências externas)."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "e", "ou",
    "com", "pra", "para", "por", "no", "na", "nos", "nas", "em", "eu", "me", "mim", "meu", "minha",
    "quero", "queria", "gostaria", "vou", "querer", "pode", "ser", "algum", "alguma", "qualquer",
    "coisa", "hoje", "ai", "aí", "tipo", "bem", "muito", "mais", "menos", "so", "só", "tambem",
    "também", "que", "to", "tô", "estou", "afim", "a fim", "comer", "pedir", "pede", "manda",
    "favor", "por favor", "gente", "pessoal", "bora", "vamos", "ne", "né", "boa", "bom",
}

# Sinônimos e famílias de pratos: aproximam a linguagem do chat ao vocabulário do cardápio.
SYNONYMS: dict[str, list[str]] = {
    "hamburguer": ["burger", "burguer", "x", "smash", "lanche", "sanduiche"],
    "burger": ["hamburguer", "burguer", "smash"],
    "lanche": ["burger", "sanduiche", "pastel", "coxinha", "wrap"],
    "japones": ["sushi", "temaki", "yakissoba", "hot", "roll", "poke"],
    "japa": ["sushi", "temaki", "hot", "roll"],
    "sushi": ["japones", "temaki", "roll"],
    "pizza": ["margherita", "calabresa", "portuguesa", "calzone"],
    "massa": ["pizza", "yakissoba", "macarrao"],
    "saudavel": ["salada", "bowl", "poke", "wrap", "grelhado"],
    "leve": ["salada", "bowl", "poke", "wrap", "grelhado"],
    "fit": ["salada", "bowl", "poke", "grelhado"],
    "salada": ["bowl", "caesar"],
    "frango": ["chicken", "galinha"],
    "chicken": ["frango"],
    "carne": ["burger", "strogonoff", "feijoada", "burrito"],
    "mexicano": ["burrito", "tacos", "nachos"],
    "mexicana": ["burrito", "tacos", "nachos"],
    "doce": ["acai", "brownie", "sobremesa", "sorvete"],
    "sobremesa": ["acai", "brownie", "sorvete"],
    "caseira": ["prato", "feito", "marmita", "feijoada", "strogonoff", "parmegiana"],
    "caseiro": ["prato", "feito", "marmita", "feijoada", "strogonoff", "parmegiana"],
    "pf": ["prato", "feito", "marmita"],
    "marmita": ["prato", "feito"],
    "batata": ["fritas", "frita"],
    "salgado": ["coxinha", "pastel"],
}

RESTRICTION_PATTERNS: dict[str, list[str]] = {
    "vegano": [r"\bvegan[oa]s?\b", r"\bplant[ -]?based\b"],
    "vegetariano": [r"\bvegetarian[oa]s?\b", r"\bsem carne\b", r"\bnao como carne\b"],
    "sem_lactose": [r"\bsem lactose\b", r"\bintolerante a lactose\b", r"\bzero lactose\b"],
    "sem_gluten": [r"\bsem gluten\b", r"\bcelia(c|qu)[oa]\b"],
}


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    stop = {normalize(s) for s in STOPWORDS}
    return [t for t in normalize(text).replace("-", " ").split() if t not in stop and len(t) > 1 or t == "x"]


def stem(token: str) -> str:
    """Stemming leve: remove plural e gênero para casar 'pizzas'/'pizza', 'vegana'/'vegano'."""
    for suffix in ("oes", "aes", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token


def trigrams(token: str) -> set[str]:
    padded = f"  {token} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}
