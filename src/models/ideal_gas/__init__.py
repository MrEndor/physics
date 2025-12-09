import numpy as np
import streamlit as st

from models.ideal_gas.charts import (
    create_3d_particle_positions_chart,
    create_energy_evolution_chart,
    create_height_distribution_chart,
    create_temperature_evolution_chart,
    create_velocity_distribution_chart,
)
from models.ideal_gas.config import SimulationConfig
from models.ideal_gas.objects import IdealGasSimulation
from models.ideal_gas.utils import (
    calculate_barometric_formula,
    calculate_maxwell_distribution,
)


def page() -> None:  # noqa: PLR0915, PLR0914
    st.set_page_config(
        page_title="М7Б. Статистика идеального газа со столкновениями частиц",
        layout="wide",
    )
    st.title("М7Б. Статистика идеального газа со столкновениями частиц")

    with st.sidebar:
        st.header("Параметры симуляции")

        num_particles = st.slider(
            "Количество молекул",
            min_value=50,
            max_value=400,
            value=150,
            step=50,
            help=(
                "Меньше частиц - быстрее симуляция. "
                "100–200 обычно достаточно, чтобы увидеть распределения."
            ),
        )

        container_height = st.slider(
            "Высота сосуда (м)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.5,
        )

        container_radius = st.slider(
            "Радиус сосуда (м)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.5,
        )

        initial_velocity = st.slider(
            "Начальная скорость молекул (м/с)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
        )

        simulation_time = st.slider(
            "Время симуляции (с)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Большее время - больше шагов и медленнее визуализация.",
        )

        dt = st.slider(
            "Шаг времени (мс)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.5,
            help="Крупный шаг ускоряет расчёт, но делает траектории грубее.",
        )

        use_thermostat = st.checkbox(
            "Использовать термостат",
            value=False,
        )

        if use_thermostat:
            target_temperature = st.slider(
                "Температура термостата (K)",
                min_value=100.0,
                max_value=800.0,
                value=300.0,
                step=50.0,
            )
        else:
            target_temperature = None

    config = SimulationConfig(
        num_particles=num_particles,
        container_height=container_height,
        container_radius=container_radius,
        initial_velocity=initial_velocity,
        simulation_time=simulation_time,
        dt=dt / 1000.0,
        use_thermostat=use_thermostat,
        target_temperature=target_temperature,
    )

    if st.button("Запустить симуляцию", use_container_width=True):
        with st.spinner("Выполняется симуляция молекулярной динамики..."):
            simulation = IdealGasSimulation(config)

            positions_history, velocities_history, times = simulation.run()

            final_positions = positions_history[-1]
            final_velocities = velocities_history[-1]

            kinetic_energy = (
                0.5
                * config.particle_mass
                * np.sum(final_velocities**2)
                / config.num_particles
            )
            temperature = 2.0 * kinetic_energy / (3.0 * config.k_B)

            max_speed = np.max(np.linalg.norm(final_velocities, axis=1))
            maxwell_speeds = np.linspace(0, max_speed * 1.2, 100)
            maxwell_dist = calculate_maxwell_distribution(
                maxwell_speeds, temperature, config.particle_mass
            )

            heights = np.linspace(0, config.container_height, 100)
            scale_height = (
                config.k_B * temperature / (config.particle_mass * config.g)
            )
            barometric_dist = calculate_barometric_formula(
                heights, scale_height
            )

            st.success("Симуляция завершена")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Определенная температура", f"{temperature:.1f} K")
            with col2:
                st.metric("Масштаб высоты", f"{scale_height:.3f} м")
            with col3:
                mean_height = np.mean(final_positions[:, 2])
                st.metric("Средняя высота", f"{mean_height:.3f} м")
            with col4:
                mean_speed = np.mean(np.linalg.norm(final_velocities, axis=1))
                st.metric("Средняя скорость", f"{mean_speed:.2f} м/с")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Распределение скоростей",
                "Распределение высот",
                "Эволюция энергии",
                "Эволюция температуры",
                "3D позиции частиц",
            ])

            with tab1:
                st.subheader("Распределение Максвелла по скоростям")
                chart = create_velocity_distribution_chart(
                    final_velocities,
                    maxwell_speeds,
                    maxwell_dist,
                    temperature,
                )
                st.altair_chart(chart, use_container_width=True)

            with tab2:
                st.subheader("Барометрическое распределение по высоте")
                chart = create_height_distribution_chart(
                    final_positions[:, 2],
                    heights,
                    barometric_dist,
                    scale_height,
                )
                st.altair_chart(chart, use_container_width=True)

            with tab3:
                st.subheader("Эволюция энергии системы")
                chart = create_energy_evolution_chart(
                    positions_history, velocities_history, times, config
                )
                st.altair_chart(chart, use_container_width=True)

            with tab4:
                st.subheader("Эволюция температуры")
                chart = create_temperature_evolution_chart(
                    velocities_history, times, config
                )
                st.altair_chart(chart, use_container_width=True)

            with tab5:
                st.subheader("3D позиции молекул в конечном состоянии")
                chart = create_3d_particle_positions_chart(
                    final_positions, config
                )
                st.altair_chart(chart, use_container_width=True)

            st.divider()
