from dataclasses import dataclass

import numpy as np
import streamlit as st
from scipy import integrate

from models.pendulum.config import PHYSICAL_CONSTANTS


@dataclass
class PendulumParameters:
    length: float = 1.0
    mass: float = 1.0
    pendulum_type: str = "uniform_rod"
    friction_coeff: float = 0.0
    phi0_deg: float = 30.0
    omega0: float = 0.0
    t_max: float = 20.0
    dt: float = 0.01


class PendulumModel:
    def __init__(self) -> None:
        self.g = PHYSICAL_CONSTANTS["g"]

    def _get_pendulum_properties(
        self, params: PendulumParameters
    ) -> tuple[float, float, float]:
        length = params.length
        mass = params.mass

        if params.pendulum_type == "point_mass":
            i_o = mass * length**2
            cm_distance = length
        elif params.pendulum_type == "uniform_rod":
            i_o = mass * length**2 / 3
            cm_distance = length / 2
        else:
            i_o = mass * length**2 / 3
            cm_distance = length / 2

        omega0 = np.sqrt(mass * self.g * cm_distance / i_o)

        return i_o, cm_distance, omega0

    def _equations_of_motion(
        self, t: float, state: np.ndarray, params: PendulumParameters
    ) -> np.ndarray:
        phi, omega = state

        _i_o, _cm_distance, omega0 = self._get_pendulum_properties(params)

        beta = params.friction_coeff / 2

        dphi_dt = omega
        domega_dt = -(omega0**2) * np.sin(phi) - 2 * beta * omega

        return np.array([dphi_dt, domega_dt])

    @st.cache_data(ttl=300)
    def simulate_motion(
        _self,  # noqa: N805
        params: PendulumParameters,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi0 = np.radians(params.phi0_deg)
        omega0 = params.omega0
        initial_state = np.array([phi0, omega0])

        t_eval = np.arange(0, params.t_max, params.dt)

        solution = integrate.solve_ivp(
            fun=lambda t, state: _self._equations_of_motion(t, state, params),  # noqa: SLF001
            t_span=(0, params.t_max),
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        phi = solution.y[0]
        omega = solution.y[1]
        t = solution.t

        return phi, omega, t

    def calculate_energy(
        self, phi: np.ndarray, omega: np.ndarray, params: PendulumParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i_o, cm_distance, _ = self._get_pendulum_properties(params)
        mass = params.mass

        kinetic = 0.5 * i_o * omega**2

        potential = mass * self.g * cm_distance * (1 - np.cos(phi))

        total = kinetic + potential

        return kinetic, potential, total

    def calculate_period(self, phi: np.ndarray, t: np.ndarray) -> float:  # noqa: PLR6301
        min_samples = 100
        min_correlation_threshold = 0.5
        min_period_seconds = 0.5
        min_zero_crossings = 4

        if len(phi) < min_samples:
            return 0.0

        try:
            phi_centered = phi - np.mean(phi)

            correlation = np.correlate(phi_centered, phi_centered, mode="full")
            correlation = correlation[correlation.size // 2 :]

            correlation /= correlation[0]

            dt = t[1] - t[0] if len(t) > 1 else 0.01
            min_period_samples = int(min_period_seconds / dt)

            if len(correlation) <= min_period_samples:
                return 0.0

            peaks = [
                i
                for i in range(min_period_samples, len(correlation) - 1)
                if (
                    correlation[i - 1] < correlation[i] > correlation[i + 1]
                    and correlation[i] > min_correlation_threshold
                )
            ]

            if peaks:
                period_samples = peaks[0]
                period = period_samples * dt
                return float(period)

        except Exception:  # noqa: S110
            pass

        zero_crossings = []

        for i in range(1, len(phi)):
            if phi[i - 1] * phi[i] < 0 and phi[i] != phi[i - 1]:
                dt_interp = (
                    -phi[i - 1] * (t[i] - t[i - 1]) / (phi[i] - phi[i - 1])
                )
                t_crossing = t[i - 1] + dt_interp
                zero_crossings.append(t_crossing)

        if len(zero_crossings) >= min_zero_crossings:
            periods = [
                zero_crossings[i + 2] - zero_crossings[i]
                for i in range(0, len(zero_crossings) - 2, 2)
            ]

            if periods:
                return float(np.mean(periods))

        return 0.0
