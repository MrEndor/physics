from dataclasses import dataclass
from operator import itemgetter
from typing import Any

import numpy as np

from models.billiard.config import PHYSICAL_CONSTANTS
from models.billiard.objects import BilliardParameters

MIN_TIME_POINTS = 2


@dataclass
class CollisionMetrics:
    times: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    params: BilliardParameters


def calculate_collision_metrics(metrics: CollisionMetrics) -> dict[str, Any]:
    distances = np.sqrt(
        (metrics.x2 - metrics.x1) ** 2 + (metrics.y2 - metrics.y1) ** 2
    )
    min_distance = np.min(distances)
    min_distance_index = np.argmin(distances)
    min_distance_time = metrics.times[min_distance_index]

    collision_threshold = 2 * metrics.params.ball_radius
    collision_indices = np.where(distances <= collision_threshold)[0]

    collision_time = None
    collision_detected = len(collision_indices) > 0

    if collision_detected:
        collision_time = metrics.times[collision_indices[0]]

    speeds = _calculate_speeds(metrics)

    return {
        "collision_time": collision_time,
        "collision_detected": collision_detected,
        "min_distance": min_distance,
        "min_distance_time": min_distance_time,
        **speeds,
    }


def _calculate_speeds(metrics: CollisionMetrics) -> dict[str, Any]:
    if len(metrics.times) > 1:
        dt = metrics.times[1] - metrics.times[0]

        vx1 = np.gradient(metrics.x1, dt)
        vy1 = np.gradient(metrics.y1, dt)
        vx2 = np.gradient(metrics.x2, dt)
        vy2 = np.gradient(metrics.y2, dt)

        speeds1 = np.sqrt(vx1**2 + vy1**2)
        speeds2 = np.sqrt(vx2**2 + vy2**2)

        return {
            "initial_cue_speed": speeds1[0],
            "initial_target_speed": speeds2[0],
            "final_cue_speed": speeds1[-1],
            "final_target_speed": speeds2[-1],
            "max_cue_speed": np.max(speeds1),
            "max_target_speed": np.max(speeds2),
        }

    return {
        "initial_cue_speed": np.sqrt(
            metrics.params.cue_vx**2 + metrics.params.cue_vy**2
        ),
        "initial_target_speed": np.sqrt(
            metrics.params.target_vx**2 + metrics.params.target_vy**2
        ),
        "final_cue_speed": 0,
        "final_target_speed": 0,
        "max_cue_speed": 0,
        "max_target_speed": 0,
    }


@dataclass
class PhysicsAnalysis:
    times: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    params: BilliardParameters
    metrics: dict[str, Any]


def analyze_collision_physics(data: PhysicsAnalysis) -> dict[str, Any]:
    if len(data.times) < MIN_TIME_POINTS:
        return {"error": "Недостаточно данных для анализа"}

    dt = data.times[1] - data.times[0]

    velocities = _calculate_velocities(data, dt)
    accelerations = _calculate_accelerations(velocities, dt)
    energies = _calculate_energies(velocities, data.params.ball_mass)
    momentum = _calculate_momentum(velocities, data.params.ball_mass)

    conservation_analysis = _analyze_conservation_laws(energies, momentum)
    collision_analysis = _analyze_collision_details(data, velocities, dt)
    trajectory_analysis = _analyze_trajectories(data, velocities, accelerations)

    return {
        "conservation_laws": conservation_analysis,
        "collision": collision_analysis,
        "trajectory": trajectory_analysis,
        "time_series": {
            "times": data.times,
            "kinetic_energy": energies,
            "momentum": momentum,
            "velocities": {
                "cue": np.sqrt(velocities["vx1"] ** 2 + velocities["vy1"] ** 2),
                "target": np.sqrt(
                    velocities["vx2"] ** 2 + velocities["vy2"] ** 2
                ),
            },
            "accelerations": {
                "cue": accelerations["a1"],
                "target": accelerations["a2"],
            },
        },
    }


def _calculate_velocities(data: PhysicsAnalysis, dt: float) -> dict[str, Any]:
    return {
        "vx1": np.gradient(data.x1, dt),
        "vy1": np.gradient(data.y1, dt),
        "vx2": np.gradient(data.x2, dt),
        "vy2": np.gradient(data.y2, dt),
    }


def _calculate_accelerations(
    velocities: dict[str, Any], dt: float
) -> dict[str, Any]:
    ax1 = np.gradient(velocities["vx1"], dt)
    ay1 = np.gradient(velocities["vy1"], dt)
    ax2 = np.gradient(velocities["vx2"], dt)
    ay2 = np.gradient(velocities["vy2"], dt)

    return {"a1": np.sqrt(ax1**2 + ay1**2), "a2": np.sqrt(ax2**2 + ay2**2)}


def _calculate_energies(velocities: dict[str, Any], mass: float) -> np.ndarray:
    v1 = np.sqrt(velocities["vx1"] ** 2 + velocities["vy1"] ** 2)
    v2 = np.sqrt(velocities["vx2"] ** 2 + velocities["vy2"] ** 2)

    ke1 = 0.5 * mass * v1**2
    ke2 = 0.5 * mass * v2**2

    value: np.ndarray = ke1 + ke2

    return value


def _calculate_momentum(velocities: dict[str, Any], mass: float) -> np.ndarray:
    px_total = mass * (velocities["vx1"] + velocities["vx2"])
    py_total = mass * (velocities["vy1"] + velocities["vy2"])

    value: np.ndarray = np.sqrt(px_total**2 + py_total**2)
    return value


def _analyze_conservation_laws(
    energies: np.ndarray, momentum: np.ndarray
) -> dict[str, Any]:
    initial_ke = energies[0]
    final_ke = energies[-1]
    energy_loss = initial_ke - final_ke
    energy_conservation_ratio = final_ke / initial_ke if initial_ke > 0 else 1.0

    initial_momentum = momentum[0]
    final_momentum = momentum[-1]
    momentum_change = abs(final_momentum - initial_momentum)
    momentum_conservation_ratio = (
        final_momentum / initial_momentum if initial_momentum > 0 else 1.0
    )

    return {
        "initial_energy": initial_ke,
        "final_energy": final_ke,
        "energy_loss": energy_loss,
        "energy_conservation_ratio": energy_conservation_ratio,
        "initial_momentum": initial_momentum,
        "final_momentum": final_momentum,
        "momentum_change": momentum_change,
        "momentum_conservation_ratio": momentum_conservation_ratio,
    }


def _analyze_collision_details(
    data: PhysicsAnalysis, velocities: dict[str, Any], dt: float
) -> dict[str, Any]:
    if not data.metrics["collision_detected"]:
        return {
            "momentum_transfer": 0.0,
            "energy_transfer_efficiency": 0.0,
            "cue_energy_change": 0.0,
            "target_energy_change": 0.0,
            "collision_force_estimate": 0.0,
        }

    collision_idx = np.argmin(
        np.abs(data.times - data.metrics["collision_time"])
    )

    before_idx = max(0, collision_idx - 10)
    after_idx = min(len(data.times) - 1, collision_idx + 10)

    momentum_transfer_x = data.params.ball_mass * (
        velocities["vx2"][after_idx] - velocities["vx2"][before_idx]
    )
    momentum_transfer_y = data.params.ball_mass * (
        velocities["vy2"][after_idx] - velocities["vy2"][before_idx]
    )
    momentum_transfer = np.sqrt(momentum_transfer_x**2 + momentum_transfer_y**2)

    v1_before = np.sqrt(
        velocities["vx1"][before_idx] ** 2 + velocities["vy1"][before_idx] ** 2
    )
    v1_after = np.sqrt(
        velocities["vx1"][after_idx] ** 2 + velocities["vy1"][after_idx] ** 2
    )
    v2_after = np.sqrt(
        velocities["vx2"][after_idx] ** 2 + velocities["vy2"][after_idx] ** 2
    )

    ke1_before = 0.5 * data.params.ball_mass * v1_before**2
    ke1_after = 0.5 * data.params.ball_mass * v1_after**2
    ke2_after = 0.5 * data.params.ball_mass * v2_after**2

    ke_change_cue = ke1_after - ke1_before
    ke_change_target = ke2_after

    energy_transfer_efficiency = (
        ke_change_target / ke1_before if ke1_before > 0 else 0.0
    )

    return {
        "momentum_transfer": momentum_transfer,
        "energy_transfer_efficiency": energy_transfer_efficiency,
        "cue_energy_change": ke_change_cue,
        "target_energy_change": ke_change_target,
        "collision_force_estimate": momentum_transfer / (dt * 20),
    }


def _analyze_trajectories(
    data: PhysicsAnalysis,
    velocities: dict[str, Any],
    accelerations: dict[str, Any],
) -> dict[str, Any]:
    cue_distance = np.sum(
        np.sqrt(np.diff(data.x1) ** 2 + np.diff(data.y1) ** 2)
    )
    target_distance = np.sum(
        np.sqrt(np.diff(data.x2) ** 2 + np.diff(data.y2) ** 2)
    )

    v1 = np.sqrt(velocities["vx1"] ** 2 + velocities["vy1"] ** 2)
    v2 = np.sqrt(velocities["vx2"] ** 2 + velocities["vy2"] ** 2)

    return {
        "cue_distance_traveled": cue_distance,
        "target_distance_traveled": target_distance,
        "max_cue_acceleration": np.max(accelerations["a1"]),
        "max_target_acceleration": np.max(accelerations["a2"]),
        "time_at_max_speed_cue": data.times[np.argmax(v1)],
        "time_at_max_speed_target": data.times[np.argmax(v2)],
    }


@dataclass
class WallCollisionData:
    times: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    ball_radius: float
    table_length: float
    table_width: float


def detect_wall_collisions(data: WallCollisionData) -> list[Any]:
    return _detect_collisions(data)


def _detect_collisions(data: WallCollisionData) -> list[Any]:
    wall_collisions = []

    wall_collisions.extend(
        _check_wall_collision(data.x1, data.y1, "биток", data)
    )
    wall_collisions.extend(
        _check_wall_collision(data.x2, data.y2, "прицельный", data)
    )

    wall_collisions.sort(key=itemgetter("time"))

    return wall_collisions


def _check_wall_collision(
    x: np.ndarray, y: np.ndarray, ball_name: str, data: WallCollisionData
) -> list[Any]:
    collisions = []

    for i in range(1, len(data.times)):
        hit_left = x[i] <= data.ball_radius and x[i - 1] > data.ball_radius
        hit_right = (
            x[i] >= data.table_length - data.ball_radius
            and x[i - 1] < data.table_length - data.ball_radius
        )
        hit_bottom = y[i] <= data.ball_radius and y[i - 1] > data.ball_radius
        hit_top = (
            y[i] >= data.table_width - data.ball_radius
            and y[i - 1] < data.table_width - data.ball_radius
        )

        wall_info = [
            (hit_left, "левая"),
            (hit_right, "правая"),
            (hit_bottom, "нижняя"),
            (hit_top, "верхняя"),
        ]

        for hit, wall_name in wall_info:
            if hit:
                collisions.append({
                    "time": data.times[i],
                    "ball": ball_name,
                    "wall": wall_name,
                    "position": (x[i], y[i]),
                })

    return collisions


def validate_initial_positions(params: BilliardParameters) -> tuple[bool, str]:
    distance = np.sqrt(
        (params.target_x - params.cue_x) ** 2
        + (params.target_y - params.cue_y) ** 2
    )

    min_distance = 2 * params.ball_radius

    if distance < min_distance:
        return (
            False,
            f"Шары перекрываются! Расстояние: {distance:.3f} м, "
            f"минимум: {min_distance:.3f} м",
        )

    table_length = PHYSICAL_CONSTANTS["table_length"]
    table_width = PHYSICAL_CONSTANTS["table_width"]
    radius = params.ball_radius

    if (
        params.cue_x - radius < 0
        or params.cue_x + radius > table_length
        or params.cue_y - radius < 0
        or params.cue_y + radius > table_width
    ):
        return False, "Биток находится вне стола!"

    if (
        params.target_x - radius < 0
        or params.target_x + radius > table_length
        or params.target_y - radius < 0
        or params.target_y + radius > table_width
    ):
        return False, "Прицельный шар находится вне стола!"

    return True, "Начальные позиции корректны"
