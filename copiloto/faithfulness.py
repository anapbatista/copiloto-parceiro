"""Verificador de fidelidade: todo número e toda citação do relatório precisam vir
das ferramentas. É o "eval" que impede o copiloto de inventar dados para o parceiro."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{3})+|\d+)(?:[.,](\d+))?(?![\w])")
CODE = re.compile(r"`[^`]*`")
QUOTE = re.compile(r"[“\"]([^”\"]{12,})[”\"]")


def _numbers_in(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _numbers_in(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(round(float(obj), 2))
    elif isinstance(obj, str):
        out |= {n for n in _parse_numbers(obj)}
    return out


def _parse_numbers(text: str) -> list[float]:
    nums = []
    for m in NUMBER.finditer(text):
        whole = m.group(1).replace(".", "")
        frac = m.group(2)
        nums.append(round(float(f"{whole}.{frac}" if frac else whole), 2))
    return nums


def _texts_in(obj) -> list[str]:
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _texts_in(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _texts_in(v)]
    return [obj] if isinstance(obj, str) else []


@dataclass
class FaithfulnessResult:
    numbers_checked: int = 0
    unsupported_numbers: list[float] = field(default_factory=list)
    quotes_checked: int = 0
    unsupported_quotes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported_numbers and not self.unsupported_quotes


ALWAYS_OK = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 24.0, 100.0}  # contagens/listas e escalas comuns ("nota 1 a 5", "3 meses")


def check(report: str, tool_outputs: list[dict], allowed_extra: set[float] | None = None) -> FaithfulnessResult:
    evidence = set().union(*(_numbers_in(o) for o in tool_outputs)) if tool_outputs else set()
    evidence |= ALWAYS_OK | (allowed_extra or set())
    corpus = " ".join(_texts_in(tool_outputs))
    res = FaithfulnessResult()
    text_wo_quotes = CODE.sub(" ", QUOTE.sub(" ", report))  # identificadores em `código` não são métricas
    for n in _parse_numbers(text_wo_quotes):
        res.numbers_checked += 1
        if n not in evidence:
            res.unsupported_numbers.append(n)
    for q in QUOTE.findall(report):
        res.quotes_checked += 1
        core = q.strip().rstrip(".…").strip()
        if core[:80] not in corpus:
            res.unsupported_quotes.append(q[:60])
    return res


def dumps(outputs: list[dict]) -> str:
    return json.dumps(outputs, ensure_ascii=False)
