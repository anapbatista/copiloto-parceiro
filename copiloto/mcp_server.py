"""Servidor MCP do Copiloto: expõe as ferramentas analíticas para qualquer agente
(Claude Desktop, Cursor, agentes próprios). Execução: python -m copiloto.mcp_server"""

from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from .agent import Copilot
from .analytics import Analytics
from .data import load
from .model import ReviewClassifier

server = MCPServer("copiloto-parceiro")


@lru_cache(maxsize=1)
def analytics() -> Analytics:
    return Analytics.build(load(), ReviewClassifier.load())


@server.tool(description="Números gerais do parceiro: nota média, % negativas, % atrasos e tendência.")
def perfil_parceiro(seller_id: str) -> dict:
    return analytics().perfil_parceiro(seller_id)


@server.tool(description="Motivos das reclamações (nota até 3) com percentual e ação recomendada.")
def motivos_reclamacao(seller_id: str) -> dict:
    return analytics().motivos_reclamacao(seller_id)


@server.tool(description="Trechos literais de reclamações de um motivo (atraso, produto_errado, embalagem...).")
def exemplos(seller_id: str, motivo: str, k: int = 3) -> dict:
    return analytics().exemplos(seller_id, motivo, k)


@server.tool(description="Compara o parceiro com a mediana dos parceiros da mesma categoria.")
def comparar_com_categoria(seller_id: str) -> dict:
    return analytics().comparar_com_categoria(seller_id)


@server.tool(description="Plano de ação completo e verificado (números e citações checados contra os dados).")
def plano_de_acao(seller_id: str) -> str:
    return Copilot(analytics()).run(seller_id).report


if __name__ == "__main__":
    server.run("stdio")
