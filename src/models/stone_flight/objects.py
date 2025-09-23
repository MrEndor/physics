from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
from scipy import integrate

from src.models.stone_flight.config import PHYSICAL_CONSTANTS, DragModel


@dataclass
class FlightParameters:
    v0: float = 30.0
    angle_deg: float = 45.0
    mass: float = 1.0
    radius: float = 0.05
    t_max: float = 20.0
    dt: float = 0.01
    cd_linear: float = 0.1
    cd_quadratic: float = 0.1


class StoneFlightModel:
    def __init__(self) -> None:
        self.g = PHYSICAL_CONSTANTS["g"]
        self.rho_air = PHYSICAL_CONSTANTS["rho_air"]

    def _calculate_drag_force(
        self,
        vx: float,
        vy: float,
        drag_model: DragModel,
        mass: float,
        area: float,
    ) -> tuple[float, float]:
        velocity = np.sqrt(vx**2 + vy**2)

        if drag_model == DragModel.WITHOUT:
            return 0.0, 0.0

        if drag_model == DragModel.LINEAR:
            drag_force = self.cd_linear * velocity
        else:
            drag_force = (
                0.5 * self.rho_air * self.cd_quadratic * area * velocity**2
            )

        if velocity > 0:
            drag_x = -drag_force * vx / velocity
            drag_y = -drag_force * vy / velocity
            return drag_x, drag_y

        return 0.0, 0.0

    def _equations_of_motion(
        self,
        t: float,
        state: np.ndarray,
        drag_model: DragModel,
        mass: float,
        area: float,
    ) -> np.ndarray:
        _, _, vx, vy = state
        drag_x, drag_y = self._calculate_drag_force(
            vx, vy, drag_model, mass, area
        )

        return np.array([
            vx,  # dx/dt
            vy,  # dy/dt
            drag_x / mass,  # dvx/dt
            -self.g + drag_y / mass,  # dvy/dt
        ])

    @st.cache_data(ttl=300)
    def calculate_trajectory(
        _self,  # noqa: N805
        params: FlightParameters,
        drag_model: DragModel,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _self.cd_quadratic = params.cd_quadratic
        _self.cd_linear = params.cd_linear

        angle_rad = np.radians(params.angle_deg)
        area = np.pi * params.radius**2

        vx0 = params.v0 * np.cos(angle_rad)
        vy0 = params.v0 * np.sin(angle_rad)
        initial_state = np.array([0.0, 0.0, vx0, vy0])

        t_eval = np.arange(0, params.t_max, params.dt)

        def hit_ground(_: float, state: tuple[Any, float]) -> float:
            return state[1]

        hit_ground.terminal = True  # type: ignore[attr-defined]
        hit_ground.direction = -1  # type: ignore[attr-defined]

        solution = integrate.solve_ivp(  # type: ignore[call-overload]
            fun=lambda t, state: _self._equations_of_motion(  # noqa: SLF001
                t, state, drag_model, params.mass, area
            ),
            t_span=(0, params.t_max),
            y0=initial_state,
            t_eval=t_eval,
            events=hit_ground,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )

        return solution.y[0], solution.y[1], solution.t
