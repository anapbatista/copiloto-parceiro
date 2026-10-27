"""Provedores de catálogo. O agente depende só da interface `CatalogProvider`,
então dá para trocar o JSON local por um servidor MCP de catálogo ou uma API real
sem mexer no otimizador."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .models import MenuItem, Restaurant

DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "data" / "restaurants.json"


class CatalogProvider(Protocol):
    def list_restaurants(self) -> list[Restaurant]: ...


def restaurant_from_dict(d: dict) -> Restaurant:
    items = tuple(
        MenuItem(i["id"], i["name"], float(i["price"]), frozenset(i.get("tags", [])), i.get("description", ""))
        for i in d["items"]
    )
    return Restaurant(d["id"], d["name"], d["cuisine"], float(d["delivery_fee"]), float(d["min_order"]),
                      int(d["eta_min"]), items)


class JsonCatalog:
    def __init__(self, path: str | Path = DEFAULT_CATALOG):
        self.path = Path(path)

    def list_restaurants(self) -> list[Restaurant]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [restaurant_from_dict(r) for r in data["restaurants"]]
