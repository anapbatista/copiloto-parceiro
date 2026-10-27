"""PedeJunto: agente que transforma o caos de um pedido em grupo no WhatsApp em um
pedido otimizado em até 2 restaurantes, respeitando o orçamento privado de cada pessoa."""

from .models import Assignment, Intent, MenuItem, Participant, Plan, Restaurant
from .optimizer import OptimizerConfig, plan_group_order
from .session import GroupOrderSession

__all__ = [
    "Assignment", "Intent", "MenuItem", "Participant", "Plan", "Restaurant",
    "OptimizerConfig", "plan_group_order", "GroupOrderSession",
]
__version__ = "0.1.0"
