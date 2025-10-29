from typing import Any

import numpy as np
import streamlit as st

from models.billiard.charts import (
    create_billiard_trajectory_chart,
    create_detailed_collision_analysis_chart,
    create_velocity_analysis_chart,
)
from models.billiard.config import PHYSICAL_CONSTANTS, CollisionMethod
from models.billiard.objects import BilliardParameters, BilliardSimulation
from models.billiard.sidebar import create_billiard_sidebar, get_selected_method
from models.billiard.utils import (
    CollisionMetrics,
    PhysicsAnalysis,
    WallCollisionData,
    analyze_collision_physics,
    calculate_collision_metrics,
    detect_wall_collisions,
    validate_initial_positions,
)

HIGH_EFFICIENCY_THRESHOLD = 80
MODERATE_EFFICIENCY_THRESHOLD = 50
GOOD_ENERGY_CONSERVATION = 0.99
MODERATE_ENERGY_CONSERVATION = 0.95
EXCELLENT_MOMENTUM_CONSERVATION = 1e-4
GOOD_MOMENTUM_CONSERVATION = 1e-3


def page() -> None:
    st.set_page_config(page_title="М2. Бильярд", layout="wide")
    st.title("М2. Моделирование столкновений в бильярде")
    st.write(
        "Численное моделирование упругих столкновений в системе бильярдных "
        "шаров с использованием двух методов: законов сохранения и силовой "
        "модели Гука."
    )

    simulation = BilliardSimulation()
    params = create_billiard_sidebar()
    selected_method = get_selected_method()

    is_valid, validation_message = validate_initial_positions(params)

    if not is_valid:
        st.error(f"Ошибка в начальных условиях: {validation_message}")
        st.info("Измените позиции шаров в боковой панели")
        return

    try:
        with st.spinner("Выполняется расчет траекторий..."):
            times, x1, y1, x2, y2 = simulation.simulate_collision(
                params, selected_method
            )

        metrics = calculate_collision_metrics(
            CollisionMetrics(times, x1, y1, x2, y2, params)
        )

        physics_analysis = analyze_collision_physics(
            PhysicsAnalysis(times, x1, y1, x2, y2, params, metrics)
        )

        wall_collisions = detect_wall_collisions(
            WallCollisionData(
                times,
                x1,
                y1,
                x2,
                y2,
                params.ball_radius,
                PHYSICAL_CONSTANTS["table_length"],
                PHYSICAL_CONSTANTS["table_width"],
            )
        )

        display_simulation_results(
            times,
            x1,
            y1,
            x2,
            y2,
            params,
            selected_method,
            metrics,
            physics_analysis,
            wall_collisions,
        )

    except Exception as e:
        st.error(f"Ошибка при расчете: {e}")
        st.info("Попробуйте изменить параметры моделирования")


def display_simulation_results(
    times: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    params: BilliardParameters,
    method: CollisionMethod,
    metrics: dict[str, Any],
    physics_analysis: dict[str, Any],
    wall_collisions: list[Any],
) -> None:
    st.header("Траектории движения шаров")

    trajectory_chart = create_billiard_trajectory_chart(
        times, x1, y1, x2, y2, params, method
    )
    st.altair_chart(trajectory_chart, use_container_width=True)

    st.header("Основные результаты")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if metrics["collision_detected"]:
            st.metric(
                "Время столкновения", f"{metrics['collision_time']:.3f} с"
            )
        else:
            st.metric("Столкновение", "Не произошло")

    with col2:
        st.metric(
            "Минимальное расстояние",
            f"{metrics['min_distance']:.4f} м",
            f"в t = {metrics['min_distance_time']:.3f} с",
        )

    with col3:
        st.metric(
            "Скорость битка",
            f"Нач: {metrics['initial_cue_speed']:.2f} м/с",
            f"Кон: {metrics['final_cue_speed']:.2f} м/с",
        )

    with col4:
        st.metric(
            "Скорость прицельного",
            f"Нач: {metrics['initial_target_speed']:.2f} м/с",
            f"Кон: {metrics['final_target_speed']:.2f} м/с",
        )

    if metrics["collision_detected"]:
        display_detailed_collision_analysis(physics_analysis, metrics)
    else:
        st.info("**Анализ траектории без столкновения:**")
        min_dist = metrics["min_distance"]
        required_dist = 2 * params.ball_radius
        st.write(
            f"Шары прошли мимо друг друга. Минимальное расстояние между "
            f"центрами: {min_dist:.4f} м (требуется ≤ {required_dist:.4f} м "
            f"для столкновения)"
        )

    if wall_collisions:
        display_wall_collisions_analysis(wall_collisions)

    display_physics_analysis_charts(physics_analysis)


def display_detailed_collision_analysis(
    physics_analysis: dict[str, Any], metrics: dict[str, Any]
) -> None:
    st.header("Подробный анализ столкновения")

    collision_data = physics_analysis.get("collision", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Передача импульса")
        momentum_transfer = collision_data.get("momentum_transfer", 0)
        st.write(
            f"**Импульс, переданный прицельному шару:** "
            f"{momentum_transfer:.4f} кг⋅м/с"
        )

        force_estimate = collision_data.get("collision_force_estimate", 0)
        st.write(f"**Оценочная сила столкновения:** {force_estimate:.1f} Н")

        st.subheader("Изменение энергии")
        cue_energy_change = collision_data.get("cue_energy_change", 0)
        target_energy_change = collision_data.get("target_energy_change", 0)

        st.write(f"**Изменение энергии битка:** {cue_energy_change:.6f} Дж")
        st.write(
            f"**Изменение энергии прицельного:** {target_energy_change:.6f} Дж"
        )

    with col2:
        st.subheader("Эффективность передачи энергии")
        efficiency = collision_data.get("energy_transfer_efficiency", 0) * 100
        st.write(
            f"**Доля энергии, переданная прицельному шару:** {efficiency:.1f}%"
        )

        st.progress(min(abs(efficiency) / 100, 1.0))

        if efficiency > HIGH_EFFICIENCY_THRESHOLD:
            st.success("Высокоэффективная передача энергии")
        elif efficiency > MODERATE_EFFICIENCY_THRESHOLD:
            st.warning("Умеренная передача энергии")
        else:
            st.info("Низкая передача энергии (касательное столкновение)")


def display_wall_collisions_analysis(wall_collisions: list[Any]) -> None:
    st.header("Столкновения со стенками")

    if not wall_collisions:
        st.info("Столкновений со стенками не обнаружено")
        return

    st.write(f"**Обнаружено столкновений со стенками:** {len(wall_collisions)}")

    collision_data = []
    for i, collision in enumerate(wall_collisions):
        collision_data.append({
            "№": i + 1,
            "Время (с)": f"{collision['time']:.3f}",
            "Шар": collision["ball"],
            "Стенка": collision["wall"],
            "Позиция X (м)": f"{collision['position'][0]:.3f}",
            "Позиция Y (м)": f"{collision['position'][1]:.3f}",
        })

    if collision_data:
        st.table(collision_data)


def display_physics_analysis_charts(physics_analysis: dict[str, Any]) -> None:
    st.header("Анализ динамики системы")

    conservation_chart = create_detailed_collision_analysis_chart(
        physics_analysis
    )
    st.altair_chart(conservation_chart, use_container_width=True)

    velocity_chart = create_velocity_analysis_chart(physics_analysis)
    st.altair_chart(velocity_chart, use_container_width=True)
