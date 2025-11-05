import numpy as np
import pandas as pd
import streamlit as st

from models.pendulum.charts import (
    create_amplitude_analysis_chart,
    create_damping_analysis_chart,
    create_energy_chart,
    create_oscillation_chart,
    create_phase_portrait,
)
from models.pendulum.objects import PendulumModel, PendulumParameters
from models.pendulum.sidebar import create_pendulum_sidebar
from models.pendulum.utils import (
    analyze_energy_conservation,
    calculate_pendulum_metrics,
    theoretical_period,
)


def page() -> None:
    st.set_page_config(page_title="М5. Физический маятник", layout="wide")
    st.title("М5. Физический маятник")
    st.write(
        "Численное моделирование колебаний физического маятника в поле тяжести"
    )

    model = PendulumModel()
    params = create_pendulum_sidebar()

    analysis_type = st.sidebar.radio(
        "Тип анализа:",
        [
            "Основной анализ",
            "Анализ зависимости от амплитуды",
            "Анализ затухания",
        ],
    )

    try:
        if analysis_type == "Основной анализ":
            display_basic_analysis(model, params)
        elif analysis_type == "Анализ зависимости от амплитуды":
            display_amplitude_analysis(model, params)
        elif analysis_type == "Анализ затухания":
            display_damping_analysis(model, params)

    except Exception as e:
        st.error(f"Ошибка при моделировании: {e}")


def display_basic_analysis(  # noqa: PLR0914
    model: PendulumModel, params: PendulumParameters
) -> None:
    phi, omega, t = model.simulate_motion(params)

    if len(phi) == 0:
        st.error("Моделирование не вернуло результатов")
        return

    kinetic, potential, total = model.calculate_energy(phi, omega, params)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Колебания маятника")
        oscillation_chart = create_oscillation_chart(phi, t)
        st.altair_chart(oscillation_chart, use_container_width=True)

    with col2:
        st.subheader("Фазовый портрет")
        phase_chart = create_phase_portrait(phi, omega)
        st.altair_chart(phase_chart, use_container_width=True)

    st.subheader("Анализ энергии системы")
    energy_chart = create_energy_chart(kinetic, potential, total, t)
    st.altair_chart(energy_chart, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Основные метрики")
        metrics = calculate_pendulum_metrics(phi, omega, t)

        theoretical_t = theoretical_period(
            params.length,
            model.g,
            np.radians(params.phi0_deg),
            params.pendulum_type,
        )

        st.write("**Экспериментальные данные:**")
        st.write(
            f"• Максимальная амплитуда: {metrics['max_amplitude_deg']:.2f}°"
        )
        st.write(f"• Период колебаний: {metrics['period_s']:.3f} с")
        max_angular_vel = metrics["max_angular_velocity"]
        st.write(
            f"• Максимальная угловая скорость: {max_angular_vel:.3f} рад/с"
        )

        st.write("**Теоретические значения:**")
        st.write(f"• Теоретический период: {theoretical_t:.3f} с")
        period_diff = abs(metrics["period_s"] - theoretical_t)
        period_error = period_diff / theoretical_t * 100
        st.write(f"• Относительная ошибка: {period_error:.2f}%")

        if params.friction_coeff > 0:
            decay_rate = metrics["amplitude_decay_rate"]
            st.write(f"• Коэффициент затухания: {decay_rate:.4f} с⁻¹")

    with col4:
        st.subheader("Анализ сохранения энергии")
        energy_analysis = analyze_energy_conservation(kinetic, potential, total)

        quality = energy_analysis["conservation_quality"]
        st.write(f"**Качество сохранения энергии:** {quality}")
        st.write(
            f"• Начальная энергия: {energy_analysis['initial_energy']:.6f} Дж"
        )
        st.write(
            f"• Конечная энергия: {energy_analysis['final_energy']:.6f} Дж"
        )
        rel_change = energy_analysis["relative_change"] * 100
        st.write(f"• Относительное изменение: {rel_change:.6f}%")
        max_dev = energy_analysis["max_deviation"] * 100
        st.write(f"• Максимальное отклонение: {max_dev:.6f}%")


def display_amplitude_analysis(
    model: PendulumModel, base_params: PendulumParameters
) -> None:
    st.subheader("Зависимость периода колебаний от начальной амплитуды")

    amplitudes = [5, 10, 15, 30, 45, 60, 90, 120]

    results = {}
    progress_bar = st.progress(0)

    for i, amp in enumerate(amplitudes):
        params = PendulumParameters(**{
            **base_params.__dict__,
            "phi0_deg": amp,
            "friction_coeff": 0.0,
            "t_max": 30.0,
        })

        try:
            phi, _omega, t = model.simulate_motion(params)
            period = model.calculate_period(phi, t)

            if period > 0:
                results[amp] = period

            progress_bar.progress((i + 1) / len(amplitudes))
        except Exception as e:
            st.warning(f"Ошибка для амплитуды {amp}°: {e}")

    if results:
        chart = create_amplitude_analysis_chart(results)
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Таблица результатов")
        table_data = []

        theoretical_t0 = theoretical_period(
            base_params.length, model.g, 0, base_params.pendulum_type
        )

        for amp in sorted(results.keys()):
            period = results[amp]
            theoretical_t = theoretical_period(
                base_params.length,
                model.g,
                np.radians(amp),
                base_params.pendulum_type,
            )

            error_pct = abs(period - theoretical_t) / theoretical_t * 100
            table_data.append({
                "Амплитуда (°)": amp,
                "Период T (с)": f"{period:.4f}",
                "T/T₀": f"{period / theoretical_t0:.4f}",
                "Теория": f"{theoretical_t:.4f}",
                "Ошибка (%)": f"{error_pct:.2f}",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df)
    else:
        st.error("Не удалось получить данные для анализа")


def display_damping_analysis(  # noqa: PLR0914
    model: PendulumModel, base_params: PendulumParameters
) -> None:
    st.subheader("Анализ затухающих колебаний при различном трении")

    gamma_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    damping_data = {}
    progress_bar = st.progress(0)

    for i, gamma in enumerate(gamma_values):
        params = PendulumParameters(**{
            **base_params.__dict__,
            "friction_coeff": gamma,
            "phi0_deg": 30.0,
            "t_max": 20.0,
        })

        try:
            phi, omega, t = model.simulate_motion(params)
            damping_data[gamma] = (phi, t)

            progress_bar.progress((i + 1) / len(gamma_values))
        except Exception as e:
            st.warning(f"Ошибка для γ = {gamma}: {e}")

    if damping_data:
        chart = create_damping_analysis_chart(damping_data)
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Характеристики затухания")

        analysis_data = []
        for gamma, (phi, t) in damping_data.items():
            if gamma > 0:
                omega = np.gradient(
                    phi, t[1] - t[0] if len(t) > 1 else base_params.dt
                )
                metrics = calculate_pendulum_metrics(phi, omega, t)

                _i_o, _length_cm, omega0 = model._get_pendulum_properties(  # noqa: SLF001
                    base_params
                )
                beta = gamma / 2
                omega_d = np.sqrt(omega0**2 - beta**2) if beta < omega0 else 0
                t_damped = 2 * np.pi / omega_d if omega_d > 0 else float("inf")

                decay_rate = metrics["amplitude_decay_rate"]
                analysis_data.append({
                    "γ (с⁻¹)": gamma,
                    "β (с⁻¹)": beta,
                    "Время затухания τ (с)": f"{1 / beta:.2f}"
                    if beta > 0
                    else "∞",
                    "Период T_d (с)": f"{t_damped:.3f}"
                    if t_damped != float("inf")
                    else "∞",
                    "Коэфф. затухания (эксп.)": f"{decay_rate:.4f}",
                })

        if analysis_data:
            df = pd.DataFrame(analysis_data)
            st.dataframe(df)
    else:
        st.error("Не удалось получить данные для анализа затухания")
