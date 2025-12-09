"""М7Б. Статистика идеального газа.

Численное моделирование установления равновесия в системе молекул идеального газа
в гравитационном поле.

Основные компоненты:
- config: параметры симуляции и конфигурация
- objects: основные классы для моделирования
- charts: визуализация результатов
- utils: вспомогательные функции для обработки данных
"""

import streamlit as st
import numpy as np

from models.ideal_gas.config import SimulationConfig
from models.ideal_gas.objects import IdealGasSimulation, EquilibriumAnalyzer
from models.ideal_gas.charts import (
    create_velocity_distribution_chart,
    create_height_distribution_chart,
    create_energy_evolution_chart,
    create_3d_particle_positions_chart,
    create_temperature_evolution_chart,
)
from models.ideal_gas.utils import calculate_maxwell_distribution, calculate_barometric_formula


def page() -> None:
    st.set_page_config(page_title="М7Б. Статистика идеального газа", layout="wide")
    st.title("М7Б. Статистика идеального газа")
    st.write(
        "Численное моделирование установления равновесия в системе молекул идеального газа в гравитационном поле"
    )

    # Боковая панель с параметрами
    with st.sidebar:
        st.header("Параметры симуляции")
        st.caption("Параметры ограничены для быстрой визуализации в браузере.")

        num_particles = st.slider(
            "Количество молекул",
            min_value=50,
            max_value=400,  # было 1000
            value=150,
            step=50,
            help="Меньше частиц — быстрее симуляция. 100–200 обычно достаточно, чтобы увидеть распределения.",
        )

        container_height = st.slider(
            "Высота сосуда (м)",
            min_value=0.5,
            max_value=3.0,  # было 5.0
            value=1.5,
            step=0.5,
        )

        container_radius = st.slider(
            "Радиус сосуда (м)",
            min_value=0.5,
            max_value=2.0,  # было 3.0
            value=1.0,
            step=0.5,
        )

        initial_velocity = st.slider(
            "Начальная скорость молекул (м/с)",
            min_value=1.0,
            max_value=30.0,  # было 50.0
            value=10.0,
            step=1.0,
        )

        simulation_time = st.slider(
            "Время симуляции (с)",
            min_value=0.5,
            max_value=5.0,  # было 20.0
            value=2.0,
            step=0.5,
            help="Большее время — больше шагов и медленнее визуализация.",
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
            help="Термостат немного удорожает шаги, включайте только при необходимости.",
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
        dt=dt / 1000.0,  # Convert ms to s
        use_thermostat=use_thermostat,
        target_temperature=target_temperature,
    )

    # Запуск симуляции
    if st.button("🚀 Запустить симуляцию", use_container_width=True):
        with st.spinner("Выполняется симуляция молекулярной динамики..."):
            simulation = IdealGasSimulation(config)

            positions_history, velocities_history, times = simulation.run()

            # Анализ равновесия
            analyzer = EquilibriumAnalyzer(config)
            final_positions = positions_history[-1]
            final_velocities = velocities_history[-1]

            # Получение температуры
            kinetic_energy = 0.5 * config.particle_mass * np.sum(final_velocities**2) / config.num_particles
            temperature = 2.0 * kinetic_energy / (3.0 * 1.38e-23)  # k_B

            # Расчет теоретических распределений
            maxwell_speeds = np.linspace(0, np.max(np.linalg.norm(final_velocities, axis=1)) * 1.2, 100)
            maxwell_dist = calculate_maxwell_distribution(maxwell_speeds, temperature, config.particle_mass)

            heights = np.linspace(0, config.container_height, 100)
            scale_height = 1.38e-23 * temperature / (config.particle_mass * 9.81)
            barometric_dist = calculate_barometric_formula(heights, scale_height)

            # Выводы
            st.success("✅ Симуляция завершена!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Определенная температура", f"{temperature:.1f} K")
            with col2:
                st.metric("Масштаб высоты", f"{scale_height:.3f} м")
            with col3:
                st.metric("Средняя высота", f"{np.mean(final_positions[:, 2]):.3f} м")
            with col4:
                st.metric("Средняя скорость", f"{np.mean(np.linalg.norm(final_velocities, axis=1)):.2f} м/с")

            # Вкладки с визуализацией
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
                chart = create_energy_evolution_chart(positions_history, velocities_history, times, config)
                st.altair_chart(chart, use_container_width=True)

            with tab4:
                st.subheader("Эволюция температуры")
                chart = create_temperature_evolution_chart(velocities_history, times, config)
                st.altair_chart(chart, use_container_width=True)

            with tab5:
                st.subheader("3D позиции молекул в конечном состоянии")
                chart = create_3d_particle_positions_chart(final_positions, config)
                st.altair_chart(chart, use_container_width=True)

            # Теоретическая информация
            st.divider()
            with st.expander("📖 Теоретическая информация"):
                st.markdown("""
                ### Распределение Максвелла
                В термодинамическом равновесии распределение молекул по скоростям подчиняется распределению Максвелла:

                $$f(v) = 4\pi v^2 \left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{mv^2}{2k_B T}\right)$$

                где:
                - $m$ - масса молекулы
                - $k_B = 1.38 \times 10^{-23}$ Дж/К - постоянная Больцмана
                - $T$ - абсолютная температура

                ### Барометрическая формула
                Распределение молекул по высоте в гравитационном поле описывается барометрической формулой:

                $$n(z) = n(0) \exp\left(-\frac{mgz}{k_B T}\right)$$

                Масштаб высоты: $H_0 = \frac{k_B T}{mg}$

                ### Теорема о равнораспределении энергии
                На каждую поступательную степень свободы приходится энергия:

                $$\left\langle \frac{mv_i^2}{2} \right\rangle = \frac{1}{2}k_B T, \quad i = x, y, z$$

                Средняя полная кинетическая энергия молекулы:

                $$\left\langle E_k \right\rangle = \frac{3}{2}k_B T$$
                """)
