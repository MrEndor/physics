from dataclasses import dataclass

import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp


@dataclass
class BallParameters:
    mass: float = 1.0
    radius: float = 0.1
    mu_friction: float = 0.3
    theta_degrees: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    vx0: float = 1.0
    vy0: float = 0.0
    omega_z0: float = 0.0
    t_max: float = 10.0
    dt: float = 0.01
    epsilon: float = 1e-4


class BallModel:
    def __init__(self):
        self.g = 9.81

    def calculate_friction_force(self, vx, vy, omega_z, params):
        v_contact_x = vx - omega_z * params.radius
        v_contact_y = vy

        v_contact_magnitude = np.sqrt(v_contact_x**2 + v_contact_y**2)

        normal_force = (
            params.mass * self.g * np.cos(np.radians(params.theta_degrees))
        )

        if v_contact_magnitude < params.epsilon:
            return 0.0, 0.0, "rolling"
        max_friction_force = params.mu_friction * normal_force

        friction_x = -max_friction_force * v_contact_x / v_contact_magnitude
        friction_y = -max_friction_force * v_contact_y / v_contact_magnitude

        return friction_x, friction_y, "sliding"

    def equations_of_motion(self, t, state, params):
        _x, _y, vx, vy, omega_z = state

        gravity_x = self.g * np.sin(np.radians(params.theta_degrees))
        gravity_y = 0.0

        friction_x, friction_y, mode = self.calculate_friction_force(
            vx, vy, omega_z, params
        )

        if mode == "rolling":
            acceleration_x = (
                (5.0 / 7.0) * self.g * np.sin(np.radians(params.theta_degrees))
            )
            acceleration_y = 0.0

            angular_acceleration = acceleration_x / params.radius

        else:
            acceleration_x = gravity_x + friction_x / params.mass
            acceleration_y = gravity_y + friction_y / params.mass

            moment_of_inertia = (2.0 / 5.0) * params.mass * params.radius**2
            torque_z = friction_x * params.radius
            angular_acceleration = torque_z / moment_of_inertia

        return [vx, vy, acceleration_x, acceleration_y, angular_acceleration]

    @st.cache_data(ttl=300)
    def simulate_motion(_self, params: BallParameters):  # noqa: N805
        initial_state = [
            params.x0,
            params.y0,
            params.vx0,
            params.vy0,
            params.omega_z0,
        ]

        t_eval = np.arange(0, params.t_max, params.dt)

        solution = solve_ivp(
            fun=lambda t, state: _self.equations_of_motion(t, state, params),
            t_span=(0, params.t_max),
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        return (
            solution.y[0],
            solution.y[1],
            solution.y[2],
            solution.y[3],
            solution.y[4],
            solution.t,
        )
