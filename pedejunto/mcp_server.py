"""Servidor MCP: expõe o planejador como ferramenta para qualquer agente/LLM cliente
(Claude Desktop, Cursor, agentes próprios). Execução: `python -m pedejunto.mcp_server`.

Requer `pip install .[mcp]` (SDK oficial `mcp` >= 2)."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .catalog import JsonCatalog
from .session import GroupOrderSession

server = MCPServer("pedejunto")


@server.tool(description=(
    "Planeja um pedido em grupo em até 2 restaurantes. Recebe a lista de pessoas com o que cada uma "
    "quer comer (texto livre) e seu orçamento máximo (privado). Retorna o plano e as mensagens "
    "para o grupo e para o privado de cada pessoa."
))
def planejar_pedido_em_grupo(pessoas: list[dict]) -> dict:
    """pessoas: [{"id": "ana", "nome": "Ana", "pedido": "um burger vegano", "orcamento": 45}]"""
    session = GroupOrderSession(catalog=JsonCatalog())
    for p in pessoas:
        session.on_group_message(p["id"], p["nome"], p["pedido"])
        session.budgets[p["id"]] = float(p["orcamento"])
    result = session.close()
    return {
        "restaurantes": result.plan.restaurant_ids,
        "itens": [
            {"pessoa": a.participant_id, "restaurante": a.restaurant_id, "item": a.item.name,
             "total": a.total, "aderencia": a.match_score}
            for a in result.plan.assignments
        ],
        "sem_opcao": result.plan.unserved,
        "mensagem_grupo": result.group_text,
        "mensagens_privadas": result.private_texts,
    }


if __name__ == "__main__":
    server.run("stdio")
