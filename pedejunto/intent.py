"""Extração de intenção a partir das mensagens do grupo.

Dois parsers com a mesma interface:
- RuleBasedIntentParser: determinístico, offline, usado em testes e como fallback.
- LLMIntentParser: usa qualquer endpoint compatível com a API de chat da OpenAI
  (OpenAI, Azure, Groq, Ollama, vLLM...) com saída JSON validada; se a chamada
  falhar ou o JSON vier inválido, cai automaticamente no parser de regras.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from .models import Intent
from .text import RESTRICTION_PATTERNS, normalize, tokenize

VALID_RESTRICTIONS = set(RESTRICTION_PATTERNS)


class IntentParser(Protocol):
    def parse(self, participant_id: str, text: str) -> Intent: ...


class RuleBasedIntentParser:
    _exclusion = re.compile(r"\bsem ([a-z]+)")

    def parse(self, participant_id: str, text: str) -> Intent:
        norm = normalize(text)
        restrictions = {
            name for name, patterns in RESTRICTION_PATTERNS.items() if any(re.search(p, norm) for p in patterns)
        }
        if "vegano" in restrictions:
            restrictions.add("vegetariano")

        # "sem cebola" vira exclusão; "sem lactose/gluten/carne" já virou restrição.
        restriction_words = {"lactose", "gluten", "carne"}
        excluded = [w for w in self._exclusion.findall(norm) if w not in restriction_words]

        cleaned = self._exclusion.sub(" ", norm)
        for patterns in RESTRICTION_PATTERNS.values():
            for p in patterns:
                cleaned = re.sub(p, " ", cleaned)
        terms = [t for t in tokenize(cleaned) if not t.isdigit()]
        return Intent(participant_id, text, terms=terms, restrictions=restrictions, excluded_terms=excluded)


SYSTEM_PROMPT = """Você extrai pedidos de comida de mensagens de WhatsApp em português.
Responda APENAS com JSON no formato:
{"terms": [palavras-chave do prato, sem acento, minúsculas],
 "restrictions": [subconjunto de "vegetariano","vegano","sem_lactose","sem_gluten"],
 "excluded_terms": [ingredientes que a pessoa NÃO quer]}
Não invente pratos que a pessoa não mencionou. Gírias: "japa"=japonês, "x"=hambúrguer, "pf"=prato feito."""


class LLMIntentParser:
    def __init__(self, client, model: str, fallback: IntentParser | None = None):
        """`client` é um objeto openai.OpenAI (ou compatível)."""
        self.client = client
        self.model = model
        self.fallback = fallback or RuleBasedIntentParser()

    def parse(self, participant_id: str, text: str) -> Intent:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            )
            data = json.loads(response.choices[0].message.content)
            return self._validate(participant_id, text, data)
        except Exception:  # rede, quota, JSON inválido: degrada com elegância
            return self.fallback.parse(participant_id, text)

    @staticmethod
    def _validate(participant_id: str, text: str, data: dict) -> Intent:
        terms = [normalize(str(t)) for t in data.get("terms", []) if str(t).strip()]
        if not terms:
            raise ValueError("LLM não retornou termos")
        restrictions = {r for r in data.get("restrictions", []) if r in VALID_RESTRICTIONS}
        if "vegano" in restrictions:
            restrictions.add("vegetariano")
        excluded = [normalize(str(t)) for t in data.get("excluded_terms", [])]
        return Intent(participant_id, text, terms=terms, restrictions=restrictions, excluded_terms=excluded)
