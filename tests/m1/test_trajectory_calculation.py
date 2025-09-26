import numpy as np

from models.stone_flight.config import DragModel
from models.stone_flight.objects import FlightParameters, StoneFlightModel


def test_trajectory_shape_and_properties(
    flight_model: StoneFlightModel, basic_params: FlightParameters
) -> None:
    x, y, t = flight_model.calculate_trajectory(basic_params, DragModel.WITHOUT)

    assert len(x) == len(y) == len(t)
    assert x.shape == y.shape == t.shape
    assert abs(x[0]) < 1e-10
    assert abs(y[0]) < 1e-10
    assert t[0] == 0.0
    assert t[-1] <= basic_params.t_max


def test_no_drag_trajectory(
    flight_model: StoneFlightModel, no_drag_params: FlightParameters
) -> None:
    x, y, t = flight_model.calculate_trajectory(
        no_drag_params, DragModel.WITHOUT
    )

    v0 = no_drag_params.v0
    angle_rad = np.radians(no_drag_params.angle_deg)
    g = flight_model.g

    for i, time in enumerate(t):
        x_analytical = v0 * np.cos(angle_rad) * time
        y_analytical = v0 * np.sin(angle_rad) * time - 0.5 * g * time**2

        assert abs(x[i] - x_analytical) < 1e-5
        assert abs(y[i] - y_analytical) < 1e-5


def test_drag_effects(
    flight_model: StoneFlightModel, basic_params: FlightParameters
) -> None:
    x_no_drag, y_no_drag, _ = flight_model.calculate_trajectory(
        basic_params, DragModel.WITHOUT
    )

    x_linear, y_linear, _ = flight_model.calculate_trajectory(
        basic_params, DragModel.LINEAR
    )

    x_quad, y_quad, _ = flight_model.calculate_trajectory(
        basic_params, DragModel.QUADRATIC
    )

    assert np.max(x_linear) < np.max(x_no_drag)
    assert np.max(x_quad) < np.max(x_no_drag)
    assert np.max(y_linear) < np.max(y_no_drag)
    assert np.max(y_quad) < np.max(y_no_drag)


def test_ground_impact_detection(
    flight_model: StoneFlightModel, basic_params: FlightParameters
) -> None:
    high_precision_params = FlightParameters(
        v0=basic_params.v0,
        angle_deg=basic_params.angle_deg,
        mass=basic_params.mass,
        radius=basic_params.radius,
        t_max=basic_params.t_max,
        dt=0.0001,
        cd_linear=basic_params.cd_linear,
        cd_quadratic=basic_params.cd_quadratic,
    )

    _, y, _ = flight_model.calculate_trajectory(
        high_precision_params, DragModel.WITHOUT
    )

    assert abs(y[-1]) < 1e-3
    assert np.all(y >= -1e-3)
