import numpy as np

from models.billiard.config import PHYSICAL_CONSTANTS, CollisionMethod
from models.billiard.objects import (
    BallState,
    BilliardParameters,
    BilliardSimulation,
)


def test_wall_collision_detection(
    billiard_simulation: BilliardSimulation,
) -> None:
    ball_left = BallState(x=0.01, y=0.5, vx=1.0, vy=0.0)
    collision, wall_type = billiard_simulation._detect_wall_collision(ball_left)
    assert collision is True
    assert wall_type == "left"

    ball_right = BallState(
        x=PHYSICAL_CONSTANTS["table_length"] - 0.01, y=0.5, vx=-1.0, vy=0.0
    )
    collision, wall_type = billiard_simulation._detect_wall_collision(
        ball_right
    )
    assert collision is True
    assert wall_type == "right"

    ball_bottom = BallState(x=1.0, y=0.01, vx=0.0, vy=1.0)
    collision, wall_type = billiard_simulation._detect_wall_collision(
        ball_bottom
    )
    assert collision is True
    assert wall_type == "bottom"

    ball_top = BallState(
        x=1.0, y=PHYSICAL_CONSTANTS["table_width"] - 0.01, vx=0.0, vy=-1.0
    )
    collision, wall_type = billiard_simulation._detect_wall_collision(ball_top)
    assert collision is True
    assert wall_type == "top"

    ball_center = BallState(x=1.0, y=0.5, vx=1.0, vy=0.0)
    collision, wall_type = billiard_simulation._detect_wall_collision(
        ball_center
    )
    assert collision is False
    assert not wall_type


def test_wall_collision_reflection(
    billiard_simulation: BilliardSimulation,
) -> None:
    ball = BallState(x=0.01, y=0.5, vx=2.0, vy=1.0)
    new_ball = billiard_simulation._handle_wall_collision(ball, "left")
    assert new_ball.vx == -2.0
    assert new_ball.vy == 1.0
    assert new_ball.x == ball.radius

    ball = BallState(x=1.0, y=0.01, vx=1.0, vy=2.0)
    new_ball = billiard_simulation._handle_wall_collision(ball, "bottom")
    assert new_ball.vx == 1.0
    assert new_ball.vy == -2.0
    assert new_ball.y == ball.radius


def test_ball_collision_detection(
    billiard_simulation: BilliardSimulation,
) -> None:
    ball1 = BallState(x=1.0, y=1.0, vx=1.0, vy=0.0)
    ball2 = BallState(x=1.0 + 2 * ball1.radius, y=1.0, vx=-1.0, vy=0.0)
    assert billiard_simulation._detect_ball_collision(ball1, ball2) is True

    ball1 = BallState(x=1.0, y=1.0, vx=1.0, vy=0.0)
    ball2 = BallState(x=1.01, y=1.0, vx=-1.0, vy=0.0)
    assert billiard_simulation._detect_ball_collision(ball1, ball2) is True

    ball1 = BallState(x=1.0, y=1.0, vx=1.0, vy=0.0)
    ball2 = BallState(x=2.0, y=1.0, vx=-1.0, vy=0.0)
    assert billiard_simulation._detect_ball_collision(ball1, ball2) is False


def test_elastic_collision_conservation_laws(
    billiard_simulation: BilliardSimulation,
) -> None:
    ball1 = BallState(x=1.0, y=1.0, vx=2.0, vy=0.0, mass=1.0)
    ball2 = BallState(x=1.1, y=1.0, vx=0.0, vy=0.0, mass=1.0)

    initial_momentum_x = ball1.mass * ball1.vx + ball2.mass * ball2.vx
    initial_momentum_y = ball1.mass * ball1.vy + ball2.mass * ball2.vy
    initial_energy = 0.5 * ball1.mass * (
        ball1.vx**2 + ball1.vy**2
    ) + 0.5 * ball2.mass * (ball2.vx**2 + ball2.vy**2)

    new_ball1, new_ball2 = billiard_simulation._collision_conservation_laws(
        ball1, ball2
    )

    final_momentum_x = (
        new_ball1.mass * new_ball1.vx + new_ball2.mass * new_ball2.vx
    )
    final_momentum_y = (
        new_ball1.mass * new_ball1.vy + new_ball2.mass * new_ball2.vy
    )

    assert abs(initial_momentum_x - final_momentum_x) < 1e-10
    assert abs(initial_momentum_y - final_momentum_y) < 1e-10

    final_energy = 0.5 * new_ball1.mass * (
        new_ball1.vx**2 + new_ball1.vy**2
    ) + 0.5 * new_ball2.mass * (new_ball2.vx**2 + new_ball2.vy**2)

    assert abs(initial_energy - final_energy) < 1e-10

    assert abs(new_ball1.vx - ball2.vx) < 1e-10
    assert abs(new_ball2.vx - ball1.vx) < 1e-10


def test_simulation_trajectory_properties(
    billiard_simulation: BilliardSimulation,
    collision_params: BilliardParameters,
) -> None:
    times, x1, y1, x2, y2 = billiard_simulation.simulate_collision(
        collision_params, CollisionMethod.CONSERVATION_LAWS
    )

    assert len(times) == len(x1) == len(y1) == len(x2) == len(y2)

    assert abs(x1[0] - collision_params.cue_x) < 1e-10
    assert abs(y1[0] - collision_params.cue_y) < 1e-10
    assert abs(x2[0] - collision_params.target_x) < 1e-10
    assert abs(y2[0] - collision_params.target_y) < 1e-10

    assert times[0] == 0.0

    assert np.all(np.diff(times) >= 0)


def test_different_collision_methods_consistency(
    billiard_simulation: BilliardSimulation,
    no_collision_params: BilliardParameters,
) -> None:
    _times1, x1_cons, y1_cons, x2_cons, y2_cons = (
        billiard_simulation.simulate_collision(
            no_collision_params, CollisionMethod.CONSERVATION_LAWS
        )
    )

    _times2, x1_hooke, y1_hooke, _x2_hooke, _y2_hooke = (
        billiard_simulation.simulate_collision(
            no_collision_params, CollisionMethod.HOOKE_FORCE
        )
    )

    assert abs(x1_cons[-1] - x1_hooke[-1]) < 0.2
    assert abs(y1_cons[-1] - y1_hooke[-1]) < 0.2

    assert abs(x2_cons[-1] - no_collision_params.target_x) < 0.01
    assert abs(y2_cons[-1] - no_collision_params.target_y) < 0.01


def test_energy_conservation_in_elastic_collision(
    billiard_simulation: BilliardSimulation,
    no_collision_params: BilliardParameters,
) -> None:
    times, x1, y1, x2, y2 = billiard_simulation.simulate_collision(
        no_collision_params, CollisionMethod.CONSERVATION_LAWS
    )

    vx1 = np.gradient(x1, times)
    vy1 = np.gradient(y1, times)
    vx2 = np.gradient(x2, times)
    vy2 = np.gradient(y2, times)

    kinetic_energy = 0.5 * no_collision_params.ball_mass * (
        vx1**2 + vy1**2
    ) + 0.5 * no_collision_params.ball_mass * (vx2**2 + vy2**2)

    energy_variation = np.max(kinetic_energy) - np.min(kinetic_energy)
    initial_energy = kinetic_energy[0]

    assert energy_variation / initial_energy < 0.1


def test_billiard_parameters_validation(
    collision_params: BilliardParameters,
) -> None:
    assert collision_params.ball_mass > 0
    assert collision_params.ball_radius > 0
    assert collision_params.t_max > 0
    assert collision_params.dt > 0

    assert PHYSICAL_CONSTANTS["ball_radius"] > 0
    assert PHYSICAL_CONSTANTS["ball_mass"] > 0
    assert PHYSICAL_CONSTANTS["table_length"] > 0
    assert PHYSICAL_CONSTANTS["table_width"] > 0
