import altair as alt
import numpy as np
import pandas as pd

from models.ideal_gas.config import SimulationConfig
from models.ideal_gas.utils import (
    calculate_speed_from_velocity,
    create_statistics_dataframe,
)


def create_velocity_distribution_chart(
    velocities: np.ndarray,
    maxwell_speeds: np.ndarray,
    maxwell_dist: np.ndarray,
    temperature: float,
) -> alt.Chart:
    speeds = calculate_speed_from_velocity(velocities)

    hist, bin_edges = np.histogram(speeds, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    sim_data = pd.DataFrame({
        "speed": bin_centers,
        "probability": hist,
        "source": "Simulation",
    })

    theory_data = pd.DataFrame({
        "speed": maxwell_speeds,
        "probability": maxwell_dist,
        "source": "Maxwell",
    })

    combined_data = pd.concat([sim_data, theory_data], ignore_index=True)

    return (
        alt.Chart(combined_data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "speed:Q", title="Скорость (m/s)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y("probability:Q", title="Вероятность"),
            color=alt.Color(
                "source:N",
                title="Источник",
                scale=alt.Scale(
                    domain=["Simulation", "Maxwell"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "source:N",
                scale=alt.Scale(
                    domain=["Simulation", "Maxwell"], range=[[1, 0], [5, 5]]
                ),
            ),
        )
        .properties(
            width=600,
            height=400,
            title=f"Распределение Максвелла (T={temperature:.1f}K)",
        )
    )


def create_height_distribution_chart(
    heights: np.ndarray,
    heights_theory: np.ndarray,
    barometric_dist: np.ndarray,
    scale_height: float,
) -> alt.Chart:
    hist, bin_edges = np.histogram(heights, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    sim_data = pd.DataFrame({
        "height": bin_centers,
        "probability": hist,
        "source": "Simulation",
    })

    theory_data = pd.DataFrame({
        "height": heights_theory,
        "probability": barometric_dist,
        "source": "Barometric",
    })

    combined_data = pd.concat([sim_data, theory_data], ignore_index=True)

    return (
        alt.Chart(combined_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("height:Q", title="Высота (m)"),
            y=alt.Y("probability:Q", title="Плотность вероятности"),
            color=alt.Color(
                "source:N",
                title="Модель",
                scale=alt.Scale(
                    domain=["Simulation", "Barometric"],
                    range=["#1f77b4", "#2ca02c"],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "source:N",
                scale=alt.Scale(
                    domain=["Simulation", "Barometric"], range=[[1, 0], [5, 5]]
                ),
            ),
        )
        .properties(
            width=600,
            height=400,
            title=f"Барометрическая формула (H0={scale_height:.3f}m)",
        )
    )


def create_energy_evolution_chart(
    positions_history: np.ndarray,
    velocities_history: np.ndarray,
    times: np.ndarray,
    config: SimulationConfig,
) -> alt.Chart:
    stats_df = create_statistics_dataframe(
        positions_history, velocities_history, config.particle_mass, times
    )

    pe_initial = stats_df["potential_energy"].iloc[0]

    energy_df = pd.DataFrame({
        "time": stats_df["time"],
        "Potential": (
            stats_df["potential_energy"] / abs(pe_initial)
            if pe_initial != 0
            else stats_df["potential_energy"]
        ),
    })

    energy_melted = energy_df.melt(
        id_vars=["time"], var_name="energy_type", value_name="normalized_value"
    )

    return (
        alt.Chart(energy_melted)
        .mark_line()
        .encode(
            x=alt.X("time:Q", title="Время (s)"),
            y=alt.Y("normalized_value:Q", title="Нормализованная энергия"),
            color=alt.Color(
                "energy_type:N",
                title="Компонента",
                scale=alt.Scale(
                    domain=["Potential"],
                    range=["#ff7f0e"],
                ),
            ),
        )
        .properties(width=600, height=400, title="Эволюция энергии системы")
    )


def create_temperature_evolution_chart(
    velocities_history: np.ndarray,
    times: np.ndarray,
    config: SimulationConfig,
) -> alt.Chart:
    temps = []
    for vel_step in velocities_history:
        kinetic_per_particle = (
            0.5
            * config.particle_mass
            * np.sum(vel_step**2)
            / config.num_particles
        )
        temp = 2.0 * kinetic_per_particle / (3.0 * config.k_B)
        temps.append(temp)

    df = pd.DataFrame({"time": times, "temperature": temps})

    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("time:Q", title="Время (s)"),
            y=alt.Y("temperature:Q", title="Температура (K)"),
            color=alt.value("#d62728"),
        )
        .properties(width=600, height=400, title="Эволюция температуры")
    )


def create_3d_particle_positions_chart(
    positions: np.ndarray,
    config: SimulationConfig,
) -> alt.Chart:
    df = pd.DataFrame({
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
    })

    height_scale = alt.Scale(
        zero=False, domainMin=0, domainMax=config.container_height
    )

    return (
        alt.Chart(df)
        .mark_circle(size=50, opacity=0.6)
        .encode(
            x=alt.X("x:Q", title="X (м)", scale=alt.Scale(zero=False)),
            y=alt.Y("z:Q", title="Z - Высота (м)", scale=height_scale),
            color=alt.Color(
                "z:Q",
                title="Высота (m)",
                scale=alt.Scale(
                    scheme="viridis",
                    domainMin=0,
                    domainMax=config.container_height,
                ),
            ),
        )
        .properties(width=500, height=400, title="Позиции молекул (XZ сечение)")
    )
