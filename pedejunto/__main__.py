"""Demo de linha de comando: `python -m pedejunto` simula um grupo de WhatsApp."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .session import GroupOrderSession

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo_chat.json"


def build_parser():
    """Usa LLM se PEDEJUNTO_LLM_MODEL estiver definido (endpoint compatível com OpenAI)."""
    model = os.getenv("PEDEJUNTO_LLM_MODEL")
    if not model:
        return None
    from openai import OpenAI  # dependência opcional: pip install .[llm]

    from .intent import LLMIntentParser

    return LLMIntentParser(OpenAI(base_url=os.getenv("PEDEJUNTO_LLM_BASE_URL")), model)


def main(path: str | None = None) -> None:
    chat = json.loads(Path(path or DEMO).read_text(encoding="utf-8"))
    session = GroupOrderSession()
    if (parser := build_parser()) is not None:
        session.parser = parser

    print("=== GRUPO ===")
    for msg in chat["group"]:
        print(f"{msg['name']}: {msg['text']}")
        session.on_group_message(msg["id"], msg["name"], msg["text"])
    print("\n=== PRIVADO (só o bot vê) ===")
    for msg in chat["private"]:
        print(f"[{msg['id']} -> bot] {msg['text']}")
        session.on_private_message(msg["id"], msg["text"])

    result = session.close()
    print("\n=== BOT NO GRUPO ===")
    print(result.group_text)
    print("\n=== BOT NO PRIVADO ===")
    for pid, text in result.private_texts.items():
        print(f"[bot -> {pid}]\n{text}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
