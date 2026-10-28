"""O copiloto: gera um plano de ação para o parceiro a partir das ferramentas.

Dois modos com a mesma saída (relatório + evidências):
- Determinístico (padrão, offline): chama as ferramentas numa ordem fixa e preenche um template.
- LLM: um agente com *tool calling* (API compatível com OpenAI: OpenAI, Groq, Gemini,
  OpenRouter, Ollama...) decide quais ferramentas chamar e redige o texto.
Nos dois casos o relatório passa pelo verificador de fidelidade antes de ser entregue;
se o LLM inventar número ou citação, o copiloto devolve o relatório determinístico.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .analytics import TOOL_SCHEMAS, Analytics
from .faithfulness import FaithfulnessResult, check


def fmt(x) -> str:
    if x is None:
        return "n/d"
    return str(x).replace(".", ",")  # sem re-arredondar: o texto mostra exatamente o que a ferramenta calculou


@dataclass
class CopilotResult:
    report: str
    evidence: list[dict]
    faithfulness: FaithfulnessResult
    mode: str
    tool_calls: list[str] = field(default_factory=list)


def deterministic_report(an: Analytics, seller_id: str) -> tuple[str, list[dict], list[str]]:
    perfil = an.perfil_parceiro(seller_id)
    motivos = an.motivos_reclamacao(seller_id)
    bench = an.comparar_com_categoria(seller_id)
    evidence, calls = [perfil, motivos, bench], ["perfil_parceiro", "motivos_reclamacao", "comparar_com_categoria"]

    lines = [
        f"# Copiloto do Parceiro · `{seller_id[:8]}`",
        "",
        f"**Categoria:** {perfil['categoria_principal']} · **Período:** {perfil['periodo']} · "
        f"**Avaliações com texto:** {perfil['avaliacoes_com_texto']}",
        "",
        "## Como você está",
        f"- Nota média **{fmt(perfil['nota_media'])}**; **{fmt(perfil['pct_negativas'])}%** das avaliações são negativas "
        f"(mediana da categoria: {fmt(bench['pct_negativas_mediana_categoria'])}%).",
        f"- **{fmt(perfil['pct_pedidos_atrasados'])}%** dos pedidos chegaram depois do prazo prometido "
        f"(mediana da categoria: {fmt(bench['pct_atraso_mediana_categoria'])}%).",
    ]
    recent, before = perfil["pct_negativas_ultimos_3_meses"], perfil["pct_negativas_antes"]
    if recent is not None and before is not None:
        trend = "piorou" if recent > before + 2 else "melhorou" if recent < before - 2 else "ficou estável"
        lines.append(f"- Nos últimos 3 meses a taxa de negativas {trend}: {fmt(recent)}% (antes: {fmt(before)}%).")
    if perfil["alertas_nota_alta_texto_negativo"]:
        lines.append(f"- O modelo achou **{perfil['alertas_nota_alta_texto_negativo']}** avaliações com nota alta mas texto "
                     "negativo: reclamações que a nota esconde.")

    top = motivos["motivos"][:3]
    lines += ["", "## O que mais gera reclamação (nota até 3)"]
    if not top:
        lines.append("- Nenhum motivo recorrente identificado nas reclamações.")
    used: set[str] = set()
    for i, m in enumerate(top, 1):
        ex = an.exemplos(seller_id, m["motivo"], k=5)
        evidence.append(ex)
        calls.append("exemplos")
        lines.append(f"{i}. **{m['descricao'].capitalize()}**: {fmt(m['pct_das_reclamacoes'])}% das reclamações "
                     f"({m['ocorrencias']} ocorrências).")
        quote = next((e["texto"] for e in ex["exemplos"] if e["texto"] not in used), None)  # um trecho diferente por motivo
        if quote:
            used.add(quote)
            lines.append(f"   > “{quote}”")
        lines.append(f"   - **Ação:** {m['acao_recomendada']}")

    lines += ["", "## Prioridade da semana"]
    if top:
        lines.append(f"Atacar **{top[0]['descricao']}** primeiro: é o motivo mais frequente "
                     f"({fmt(top[0]['pct_das_reclamacoes'])}% das reclamações).")
    else:
        lines.append("Manter o padrão atual e acompanhar a tendência mensal.")
    return "\n".join(lines), evidence, calls


SYSTEM_PROMPT = """Você é o Copiloto do Parceiro: ajuda lojistas de um marketplace a reduzir reclamações.
Use SOMENTE números e trechos que vierem das ferramentas; nunca estime nem arredonde de outro jeito.
Chame perfil_parceiro, motivos_reclamacao e comparar_com_categoria; use exemplos para os principais motivos.
Escreva em português, direto, em Markdown, com: diagnóstico, 3 prioridades com ação concreta e um trecho literal
entre aspas “ ” por prioridade."""


class Copilot:
    def __init__(self, analytics: Analytics, client=None, model: str | None = None, max_steps: int = 8):
        self.an = analytics
        self.client = client
        self.model = model
        self.max_steps = max_steps

    def run(self, seller_id: str) -> CopilotResult:
        report, evidence, calls = deterministic_report(self.an, seller_id)
        base = CopilotResult(report, evidence, check(report, evidence), "deterministico", calls)
        if self.client is None:
            return base
        try:
            llm = self._run_llm(seller_id)
        except Exception:
            return base
        return llm if llm.faithfulness.ok else base  # LLM inventou algo: degrada com segurança

    def _run_llm(self, seller_id: str) -> CopilotResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Gere o plano de ação do parceiro {seller_id}."}]
        evidence, calls = [], []
        for _ in range(self.max_steps):
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOL_SCHEMAS, temperature=0)
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                report = msg.content or ""
                return CopilotResult(report, evidence, check(report, evidence), "llm", calls)
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                out = self.an.call(tc.function.name, **args)
                evidence.append(out)
                calls.append(tc.function.name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})
        raise RuntimeError("agente excedeu o número máximo de passos")
