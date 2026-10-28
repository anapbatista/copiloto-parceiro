import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from copiloto.agent import Copilot, deterministic_report
from copiloto.analytics import Analytics
from copiloto.faithfulness import check
from copiloto.model import ReviewClassifier, temporal_split, train
from copiloto.reasons import tag

FIXTURE = Path(__file__).parent / "fixtures" / "reviews_sample.csv"
SELLER = "4a3ca9315b744ce9f8e9374361493884"


@pytest.fixture(scope="session")
def data():
    return pd.read_csv(FIXTURE, dtype={"seller_id": str})


@pytest.fixture(scope="session")
def trained(data):
    return train(data)


@pytest.fixture(scope="session")
def analytics(data, trained):
    return Analytics.build(data, trained[0])


# ---------- regras de motivo ----------

@pytest.mark.parametrize("text,expected", [
    ("Até hoje não recebi o produto", "nao_recebido"),
    ("Pedi na cor preta e veio na cor vermelha", "produto_errado"),
    ("Chegou apenas uma cadeira, sendo que paguei duas", "incompleto"),
    ("Veio quebrado, caixa amassada", "defeito_qualidade"),
    ("Caixa amassada e mal embalado", "embalagem"),
    ("Mandei mensagem e ninguém responde", "atendimento"),
    ("Entrega muito atrasada, passou o prazo", "atraso"),
])
def test_regras(text, expected):
    assert expected in tag(text)


def test_elogio_sem_motivo():
    assert tag("Produto ótimo, recomendo!") == []


# ---------- modelo ----------

def test_divisao_temporal_sem_vazamento(data):
    tr, va, te = temporal_split(data)
    assert tr.data_compra.max() < va.data_compra.min() <= va.data_compra.max() < te.data_compra.min()


def test_modelo_aprende_algo(trained):
    _, metrics = trained
    assert metrics["sentimento_macro_f1"] > 0.5
    assert metrics["atraso_roc_auc"] > 0.7


def test_salvar_e_carregar(trained, tmp_path):
    model, _ = trained
    model.save(tmp_path / "m.joblib")
    loaded = ReviewClassifier.load(tmp_path / "m.joblib")
    texts = ["péssimo, não chegou", "amei o produto"]
    assert loaded.predict(texts).equals(model.predict(texts))


# ---------- ferramentas ----------

def test_perfil_bate_com_os_dados(analytics, data):
    p = analytics.perfil_parceiro(SELLER)
    s = data[data.seller_id == SELLER]
    assert p["avaliacoes_com_texto"] == len(s)
    assert p["nota_media"] == round(s.nota.mean(), 2)
    assert p["pct_pedidos_atrasados"] == round(100 * s.atrasado.mean(), 1)


def test_motivos_ordenados(analytics):
    m = analytics.motivos_reclamacao(SELLER)["motivos"]
    occ = [x["ocorrencias"] for x in m]
    assert occ == sorted(occ, reverse=True) and all(x["acao_recomendada"] for x in m)


def test_parceiro_inexistente(analytics):
    with pytest.raises(KeyError):
        analytics.perfil_parceiro("nao-existe")


# ---------- fidelidade ----------

def test_relatorio_deterministico_e_fiel(analytics):
    report, evidence, _ = deterministic_report(analytics, SELLER)
    assert check(report, evidence).ok


def test_verificador_pega_numero_e_citacao_inventados(analytics):
    report, evidence, _ = deterministic_report(analytics, SELLER)
    assert not check(report + "\n37,3% pediram reembolso.", evidence).ok
    assert not check(report + "\n> “Melhor loja do Brasil, entrega relâmpago!”", evidence).ok


# ---------- agente com LLM (cliente falso, sem rede) ----------

def _msg(content=None, calls=None):
    tool_calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                  for i, (n, a) in enumerate(calls or [])]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls or None))])


class ScriptedLLM:
    def __init__(self, final_text_fn):
        self.step, self.final_text_fn, self.seen = 0, final_text_fn, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.step += 1
        if self.step == 1:
            return _msg(calls=[("perfil_parceiro", {"seller_id": SELLER})])
        tool_out = json.loads(kwargs["messages"][-1]["content"])
        return _msg(content=self.final_text_fn(tool_out))


def test_agente_llm_fiel_e_aceito(analytics):
    llm = ScriptedLLM(lambda p: f"Sua nota média é {str(p['nota_media']).replace('.', ',')}.")
    result = Copilot(analytics, llm, "fake").run(SELLER)
    assert result.mode == "llm" and result.tool_calls == ["perfil_parceiro"]


def test_agente_llm_que_alucina_cai_no_deterministico(analytics):
    llm = ScriptedLLM(lambda p: "Sua nota média é 4,87 e 93,1% dos clientes amam você.")
    result = Copilot(analytics, llm, "fake").run(SELLER)
    assert result.mode == "deterministico" and result.faithfulness.ok


def test_agente_llm_com_erro_de_rede(analytics):
    class Broken:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(TimeoutError())))
    assert Copilot(analytics, Broken(), "fake").run(SELLER).mode == "deterministico"
