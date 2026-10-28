"""Motivos de reclamação: regras transparentes e auditáveis.

Cada motivo é uma lista de expressões regulares sobre o texto normalizado (minúsculo,
sem acento). Regras foram escolhidas porque (1) o dono do negócio precisa entender por que
uma avaliação caiu em "embalagem" e (2) não existe rótulo público de motivo. O motivo
"atraso" é validado contra o rótulo real de atraso da Olist em evals/run_evals.py.
"""

from __future__ import annotations

import re

from .text import normalize

RULES: dict[str, list[str]] = {
    "atraso": [
        r"\bdemorad", r"\bespera (muito|exagerad|longa)", r"\bexagerad[ao] a espera",
        r"\batras", r"\bdemor", r"fora do prazo", r"apos o prazo", r"passou (do|o) prazo", r"prazo (de entrega )?(vencido|expirado|estourado)",
        r"\bnao (chegou|foi entregue|entregaram) (no|dentro do) prazo", r"\blent[oa]\b",
    ],
    "nao_recebido": [
        r"\bn (recebi|chegou)", r"\bhj (nao|n) ",
        r"\b(nao|ainda nao|nunca) (recebi|chegou|foi entregue|entregaram|me entregaram)", r"\baguardando\b",
        r"\bate (hoje|agora|o momento) nao", r"\bcade (o|meu) (produto|pedido)",
    ],
    "produto_errado": [
        r"\bnao corresponde", r"\b(e|eh) diferente", r"\bdiferente do (produto|que|anuncio)", r"\bveio (na|no) (cor|tamanho|modelo)",
        r"\bfalsificad", r"\bparalel", r"\bnao (e|era) original", r"\b(menor|maior) (do que|que) ",
        r"\b(produto|item|cor|tamanho|modelo|marca) (errad|diferente)", r"\bveio (errad|outro|outra|diferente)",
        r"\bdiferente (do|da) (que|anunciad|foto|pedid|compr)", r"\bnao (e|era) o (que|produto)", r"\benviaram (outro|outra|errad)",
    ],
    "incompleto": [
        r"\bparcial", r"\bentregues? (apenas|somente|so)\b", r"\b(apenas|somente|so) (um|uma|1|2|dois|duas) (dos|das|de|produto|item|relogio)",
        r"\bfalt(ando|ou|a) ", r"\bincomplet", r"\b(veio|chegou|recebi) (so|somente|apenas) (um|uma|1|parte)",
        r"\b(so|somente|apenas) (um|uma|1) (dos|das|de)",
    ],
    "defeito_qualidade": [
        r"\bdefeit", r"\bquebrad", r"\b(nao|parou de) funcion", r"\bqualidade (ruim|pessima|baixa|horrivel)", r"\bdanificad",
        r"\bpessima qualidade", r"\bmal acabad", r"\bvazando", r"\brasgad", r"\btrinc",
    ],
    "embalagem": [
        r"\bembalag", r"\bcaixa (amassad|danificad|aberta|rasgad|molhad)", r"\bmal embalad", r"\bavari",
    ],
    "atendimento": [
        r"\baguardo (um )?retorno", r"\bsem retorno", r"\bnenhum retorno",
        r"\batendimento", r"\b(sem|nenhuma|nao (tive|obtive|tenho)) resposta", r"\bnao (respond|retorn)", r"\bninguem (respond|atend)",
        r"\bdescaso", r"\bsac\b", r"\bcontato com (a|o) (loja|vendedor)", r"\bfalta de respeito",
    ],
    "cobranca_preco": [
        r"\btroca", r"\bcancel",
        r"\bcobrad", r"\bcobranc", r"\breembols", r"\bestorn", r"\bdevolu", r"\bpreco (alto|abusivo)", r"\bcaro\b",
    ],
}

COMPILED = {k: [re.compile(p) for p in v] for k, v in RULES.items()}

ACTIONS: dict[str, str] = {
    "atraso": "Revisar o prazo prometido por região e o tempo de despacho; avisar o cliente proativamente quando houver risco de atraso.",
    "nao_recebido": "Acompanhar rastreio dos pedidos em aberto e acionar a transportadora antes do cliente reclamar.",
    "produto_errado": "Implantar conferência de SKU/variação na separação (foto do pacote antes de fechar) e revisar fotos/descrições do anúncio.",
    "incompleto": "Checklist de itens por pedido na expedição; destacar no anúncio o que vem na embalagem.",
    "defeito_qualidade": "Auditar lote/fornecedor dos produtos mais citados e testar amostras antes do envio.",
    "embalagem": "Reforçar embalagem dos itens frágeis (proteção interna, caixa adequada) e revisar o manuseio na coleta.",
    "atendimento": "Definir SLA de resposta (ex.: 24h) e modelo de resposta para as dúvidas mais comuns.",
    "cobranca_preco": "Agilizar reembolsos/estornos e deixar regras de troca e devolução claras no anúncio.",
}

LABELS = {
    "atraso": "atraso na entrega", "nao_recebido": "pedido não recebido", "produto_errado": "produto errado/diferente",
    "incompleto": "pedido incompleto", "defeito_qualidade": "defeito ou baixa qualidade", "embalagem": "embalagem/avaria",
    "atendimento": "atendimento do vendedor", "cobranca_preco": "cobrança, reembolso ou preço",
}


def tag(text: str) -> list[str]:
    norm = normalize(text)
    return [name for name, patterns in COMPILED.items() if any(p.search(norm) for p in patterns)]
