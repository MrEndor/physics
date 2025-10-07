from dataclasses import dataclass

import numpy as np
import streamlit as st
from scipy import integrate

from models.billiard.config import PHYSICAL_CONSTANTS, CollisionMethod

EPSILON = 1e-10


@dataclass
class BilliardParameters:
    cue_x: float = 0.5
    cue_y: float = 0.5
    cue_vx: float = 2.0
    cue_vy: float = 1.0

    target_x: float = 1.5
    target_y: float = 0.7
    target_vx: float = 0.0
    target_vy: float = 0.0

    t_max: float = 10.0
    dt: float = 0.01

    ball_mass: float = PHYSICAL_CONSTANTS["ball_mass"]
    ball_radius: float = PHYSICAL_CONSTANTS["ball_radius"]
    k_spring: float = 1e6
    friction_coeff: float = PHYSICAL_CONSTANTS["friction_coeff"]


@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float
    mass: float = PHYSICAL_CONSTANTS["ball_mass"]
    radius: float = PHYSICAL_CONSTANTS["ball_radius"]


class BilliardSimulation:
    def __init__(self) -> None:
        self.table_length = PHYSICAL_CONSTANTS["table_length"]
        self.table_width = PHYSICAL_CONSTANTS["table_width"]
        self.ball_radius = PHYSICAL_CONSTANTS["ball_radius"]

    def _detect_wall_collision(self, ball: BallState) -> tuple[bool, str]:
        collision = False
        wall_type = ""

        if ball.x - ball.radius <= 0:
            collision = True
            wall_type = "left"
        elif ball.x + ball.radius >= self.table_length:
            collision = True
            wall_type = "right"

        if ball.y - ball.radius <= 0:
            collision = True
            wall_type = "bottom"
        elif ball.y + ball.radius >= self.table_width:
            collision = True
            wall_type = "top"

        return collision, wall_type

    def _handle_wall_collision(
        self, ball: BallState, wall_type: str
    ) -> BallState:
        new_ball = BallState(
            x=ball.x,
            y=ball.y,
            vx=ball.vx,
            vy=ball.vy,
            mass=ball.mass,
            radius=ball.radius,
        )

        if wall_type in {"left", "right"}:
            new_ball.vx = -ball.vx
            if wall_type == "left":
                new_ball.x = ball.radius
            else:
                new_ball.x = self.table_length - ball.radius

        elif wall_type in {"top", "bottom"}:
            new_ball.vy = -ball.vy
            if wall_type == "bottom":
                new_ball.y = ball.radius
            else:
                new_ball.y = self.table_width - ball.radius

        return new_ball

    @staticmethod
    def _detect_ball_collision(ball1: BallState, ball2: BallState) -> bool:
        distance = np.sqrt((ball1.x - ball2.x) ** 2 + (ball1.y - ball2.y) ** 2)
        return bool(distance <= (ball1.radius + ball2.radius))

    @staticmethod
    def _collision_conservation_laws(
        ball1: BallState, ball2: BallState
    ) -> tuple[BallState, BallState]:
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance == 0:
            distance = EPSILON

        nx = dx / distance
        ny = dy / distance

        v1n = ball1.vx * nx + ball1.vy * ny
        v1t = -ball1.vx * ny + ball1.vy * nx

        v2n = ball2.vx * nx + ball2.vy * ny
        v2t = -ball2.vx * ny + ball2.vy * nx

        m1, m2 = ball1.mass, ball2.mass

        v1n_new = ((m1 - m2) * v1n + 2 * m2 * v2n) / (m1 + m2)
        v2n_new = ((m2 - m1) * v2n + 2 * m1 * v1n) / (m1 + m2)

        ball1_new = BallState(
            x=ball1.x,
            y=ball1.y,
            vx=v1n_new * nx - v1t * ny,
            vy=v1n_new * ny + v1t * nx,
            mass=ball1.mass,
            radius=ball1.radius,
        )

        ball2_new = BallState(
            x=ball2.x,
            y=ball2.y,
            vx=v2n_new * nx - v2t * ny,
            vy=v2n_new * ny + v2t * nx,
            mass=ball2.mass,
            radius=ball2.radius,
        )

        return ball1_new, ball2_new

    @staticmethod
    def _hooke_force_equations(  # noqa: PLR0914
        _: float, state: np.ndarray, params: BilliardParameters
    ) -> np.ndarray:
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        fx1, fy1, fx2, fy2 = 0.0, 0.0, 0.0, 0.0

        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        overlap = 2 * params.ball_radius - distance

        if overlap > 0:
            if distance > EPSILON:
                nx = dx / distance
                ny = dy / distance
            else:
                nx, ny = 1.0, 0.0

            force_magnitude = params.k_spring * overlap

            fx1 += force_magnitude * nx
            fy1 += force_magnitude * ny
            fx2 -= force_magnitude * nx
            fy2 -= force_magnitude * ny

        friction_force = params.friction_coeff
        fx1 -= friction_force * vx1
        fy1 -= friction_force * vy1
        fx2 -= friction_force * vx2
        fy2 -= friction_force * vy2

        return np.array([
            vx1,
            vy1,
            fx1 / params.ball_mass,
            fy1 / params.ball_mass,
            vx2,
            vy2,
            fx2 / params.ball_mass,
            fy2 / params.ball_mass,
        ])

    @st.cache_data(ttl=300)
    def simulate_collision(
        _self,  # noqa: N805
        params: BilliardParameters,
        method: CollisionMethod,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if method == CollisionMethod.CONSERVATION_LAWS:
            return _self._simulate_conservation_method(params)  # noqa: SLF001
        return _self._simulate_hooke_method(params)  # noqa: SLF001

    def _simulate_conservation_method(
        self, params: BilliardParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        times = np.arange(0, params.t_max, params.dt)
        n_steps = len(times)

        x1_arr = np.zeros(n_steps)
        y1_arr = np.zeros(n_steps)
        x2_arr = np.zeros(n_steps)
        y2_arr = np.zeros(n_steps)

        ball1 = BallState(
            x=params.cue_x,
            y=params.cue_y,
            vx=params.cue_vx,
            vy=params.cue_vy,
            mass=params.ball_mass,
        )
        ball2 = BallState(
            x=params.target_x,
            y=params.target_y,
            vx=params.target_vx,
            vy=params.target_vy,
            mass=params.ball_mass,
        )

        for i, _ in enumerate(times):
            x1_arr[i] = ball1.x
            y1_arr[i] = ball1.y
            x2_arr[i] = ball2.x
            y2_arr[i] = ball2.y

            collision1, wall1 = self._detect_wall_collision(ball1)
            if collision1:
                ball1 = self._handle_wall_collision(ball1, wall1)

            collision2, wall2 = self._detect_wall_collision(ball2)
            if collision2:
                ball2 = self._handle_wall_collision(ball2, wall2)

            if self._detect_ball_collision(ball1, ball2):
                ball1, ball2 = self._collision_conservation_laws(ball1, ball2)

            ball1.x += ball1.vx * params.dt
            ball1.y += ball1.vy * params.dt
            ball2.x += ball2.vx * params.dt
            ball2.y += ball2.vy * params.dt

            friction_factor = 1 - params.friction_coeff * params.dt
            ball1.vx *= friction_factor
            ball1.vy *= friction_factor
            ball2.vx *= friction_factor
            ball2.vy *= friction_factor

        return times, x1_arr, y1_arr, x2_arr, y2_arr

    def _simulate_hooke_method(
        self, params: BilliardParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial_state = np.array([
            params.cue_x,
            params.cue_y,
            params.cue_vx,
            params.cue_vy,
            params.target_x,
            params.target_y,
            params.target_vx,
            params.target_vy,
        ])

        t_eval = np.arange(0, params.t_max, params.dt)

        solution = integrate.solve_ivp(
            fun=lambda t, state: self._hooke_force_equations(t, state, params),
            t_span=(0, params.t_max),
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )

        x1 = solution.y[0]
        y1 = solution.y[1]
        x2 = solution.y[4]
        y2 = solution.y[5]
        times = solution.t

        return times, x1, y1, x2, y2
