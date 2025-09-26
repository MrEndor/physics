from enum import Enum
from typing import Final

PHYSICAL_CONSTANTS: Final[dict[str, float]] = {
    "g": 9.81,  # ускорение свободного падения, м/с²
    "rho_air": 1.225,  # плотность воздуха, кг/м³
}


class DragModel(Enum):
    WITHOUT = "Без сопротивления"
    LINEAR = "Вязкое трение (F ∼ v)"
    QUADRATIC = "Лобовое сопротивление (F ∼ v²)"


DRAG_MODEL_COLORS: Final[dict[DragModel, str]] = {
    DragModel.WITHOUT: "#1f77b4",  # синий
    DragModel.LINEAR: "#ff7f0e",  # оранжевый
    DragModel.QUADRATIC: "#2ca02c",  # зеленый
}
