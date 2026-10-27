import json
from types import SimpleNamespace

import pytest

from pedejunto import GroupOrderSession, OptimizerConfig, Participant, plan_group_order
from pedejunto.catalog import JsonCatalog
from pedejunto.intent import LLMIntentParser, RuleBasedIntentParser
from pedejunto.matching import match_score
from pedejunto.session import parse_budget

CATALOG = JsonCatalog().list_restaurants()
PARSER = RuleBasedIntentParser()


def plan_for(rows, **cfg):
    people = [Participant(f"p{i}", f"P{i}", budget) for i, (_, budget) in enumerate(rows)]
    intents = {p.id: PARSER.parse(p.id, text) for p, (text, _) in zip(people, rows, strict=True)}
    return people, plan_group_order(people, intents, CATALOG, OptimizerConfig(**cfg))


# ---------- intenção ----------

def test_restricoes_e_exclusoes():
    it = PARSER.parse("a", "Sou vegana, quero um hambúrguer sem cebola")
    assert it.restrictions == {"vegano", "vegetariano"}
    assert it.excluded_terms == ["cebola"]
    assert "hamburguer" in it.terms


def test_sem_lactose_nao_vira_exclusao():
    it = PARSER.parse("a", "yakissoba sem lactose")
    assert it.restrictions == {"sem_lactose"}
    assert it.excluded_terms == []


def test_restricao_e_filtro_duro():
    vegan = PARSER.parse("a", "burger vegano")
    smash = next(i for r in CATALOG for i in r.items if i.id == "bl-2")
    assert match_score(vegan, smash) == 0.0


def test_fuzzy_tolera_erro_de_digitacao():
    it = PARSER.parse("a", "piza calabreza")
    calabresa = next(i for r in CATALOG for i in r.items if i.id == "np-2")
    assert match_score(it, calabresa) > 0.5


@pytest.mark.parametrize("text,value", [("meu limite é 40", 40), ("R$ 38,50", 38.5), ("até 45 conto", 45)])
def test_parse_budget(text, value):
    assert parse_budget(text) == value


# ---------- otimizador ----------

def test_respeita_orcamento_e_limite_de_restaurantes():
    rows = [("hambúrguer vegano", 40), ("smash com bacon", 45), ("pizza margherita", 38),
            ("sushi", 70), ("açaí", 30), ("frango", 35)]
    people, plan = plan_for(rows)
    budgets = {p.id: p.budget for p in people}
    assert len({a.restaurant_id for a in plan.assignments}) <= 2
    assert all(a.total <= budgets[a.participant_id] for a in plan.assignments)


def test_um_restaurante_quando_configurado():
    _, plan = plan_for([("burger", 50), ("pizza", 50), ("batata frita", 50)], max_restaurants=1)
    assert len(plan.restaurant_ids) == 1


def test_divide_taxa_entre_quem_pediu_no_restaurante():
    _, plan = plan_for([("smash com bacon", 60), ("x burguer", 60)], max_restaurants=1)
    fee = next(r.delivery_fee for r in CATALOG if r.id == plan.restaurant_ids[0])
    assert sum(a.fee_share for a in plan.assignments) == pytest.approx(fee, abs=0.02)


def test_orcamento_impossivel_deixa_pessoa_sem_pedido():
    _, plan = plan_for([("smash com bacon", 50), ("combo sushi 16 peças", 12)])
    assert plan.unserved == ["p1"]
    assert plan.served == 1


def test_lista_vazia():
    plan = plan_group_order([], {}, CATALOG)
    assert plan.assignments == [] and plan.unserved == []


# ---------- sessão e privacidade ----------

def demo_session():
    s = GroupOrderSession()
    for pid, name, text, budget in [
        ("ana", "Ana", "hambúrguer vegano", "meu limite é 40"),
        ("bia", "Bia", "pizza margherita", "37,90"),
        ("caio", "Caio", "smash com bacon", "45"),
    ]:
        s.on_group_message(pid, name, text)
        s.on_private_message(pid, budget)
    return s


def test_mensagem_do_grupo_nao_vaza_valores():
    result = demo_session().close()
    assert "R$" not in result.group_text
    assert "37,90" not in result.group_text and "40" not in result.group_text
    assert "R$" in result.private_texts["bia"]


def test_pendencias_de_orcamento():
    s = GroupOrderSession()
    s.on_group_message("x", "Xavier", "pizza")
    assert s.pending_budgets() == ["Xavier"]


# ---------- LLM (cliente falso, sem rede) ----------

class FakeClient:
    def __init__(self, content=None, error=None):
        def create(**_):
            if error:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_llm_parser_valida_json():
    payload = {"terms": ["sushi"], "restrictions": ["sem_lactose", "inventada"], "excluded_terms": []}
    it = LLMIntentParser(FakeClient(json.dumps(payload)), "m").parse("a", "japa sem lactose")
    assert it.terms == ["sushi"] and it.restrictions == {"sem_lactose"}


def test_llm_parser_faz_fallback_em_erro():
    it = LLMIntentParser(FakeClient(error=TimeoutError()), "m").parse("a", "pizza vegana")
    assert "pizza" in it.terms and "vegano" in it.restrictions


# ---------- MCP ----------

def test_ferramenta_mcp():
    pytest.importorskip("mcp.server.mcpserver")
    from pedejunto.mcp_server import planejar_pedido_em_grupo

    out = planejar_pedido_em_grupo([
        {"id": "a", "nome": "Ana", "pedido": "temaki de salmão", "orcamento": 50},
        {"id": "b", "nome": "Bruno", "pedido": "hot roll", "orcamento": 40},
    ])
    assert out["restaurantes"] == ["sakura-sushi"]
    assert len(out["itens"]) == 2
