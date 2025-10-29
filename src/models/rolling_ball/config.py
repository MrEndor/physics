from enum import Enum
from typing import Final


class SurfaceType(Enum):
    INCLINED = "Наклонная плоскость"
    HORIZONTAL = "Горизонтальная плоскость"


class RollingMode(Enum):
    NO_SLIP = "Качение без проскальзывания"
    WITH_SLIP = "Качение с проскальзыванием"
    PURE_SLIDING = "Чистое скольжение"


PHYSICAL_CONSTANTS: Final[dict[str, float]] = {
    "g": 9.81,
    "ball_mass": 0.5,
    "ball_radius": 0.05,
    "moment_of_inertia_factor": 2 / 5,
}

DEFAULT_PARAMS: Final[dict[str, float]] = {
    "theta_deg": 30.0,
    "mu_friction": 0.3,
    "v0": 0.0,
    "omega0": 0.0,
    "x0": 0.0,
    "y0": 0.0,
    "t_max": 10.0,
    "dt": 0.01,
    "epsilon": 1e-10,
}
