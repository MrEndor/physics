from enum import Enum
from typing import Final

PHYSICAL_CONSTANTS: Final[dict[str, float]] = {
    "g": 9.81,
    "ball_radius": 0.0286,
    "ball_mass": 0.17,
    "table_length": 2.84,
    "table_width": 1.42,
    "restitution_coeff": 0.95,
    "friction_coeff": 0.01,
}


class CollisionMethod(Enum):
    CONSERVATION_LAWS = "Законы сохранения"
    HOOKE_FORCE = "Силовая модель (Гук)"


class BallType(Enum):
    CUE = "Биток"
    TARGET = "Прицельный"
    OBSTACLE = "Препятствие"


BALL_COLORS: Final[dict[BallType, str]] = {
    BallType.CUE: "#FFFFFF",  # белый
    BallType.TARGET: "#FF0000",  # красный
    BallType.OBSTACLE: "#0000FF",  # синий
}

COLLISION_METHOD_COLORS: Final[dict[CollisionMethod, str]] = {
    CollisionMethod.CONSERVATION_LAWS: "#1f77b4",  # синий
    CollisionMethod.HOOKE_FORCE: "#ff7f0e",  # оранжевый
}
