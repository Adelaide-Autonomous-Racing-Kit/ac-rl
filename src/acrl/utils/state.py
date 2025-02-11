from collections import deque
from typing import Dict, Union

from acrl.utils.look_ahead import LookAhead
import numpy as np

TOP_SPEED_MS = 80.0
CURVATURE_NORMALISATION = 0.1
# Observation Key : normalising factor
STATE_REPRESENTATION = {
    "speed_kmh": TOP_SPEED_MS,
    "throttle": 1.0,
    "brake": 1.0,
    "steering_angle": 450.0,
    "gap": 10.0,
    "final_force_feedback": 1.0,
    "rpm": 1_000.0,
    "acceleration_g_X": 0.5,
    "acceleration_g_Y": 0.5,
    "gear": 8.0,
    "local_angular_velocity_Y": np.pi,
    "local_velocity_X": TOP_SPEED_MS,
    "local_velocity_Y": 10.0,
    "wheel_slip_front_left": 25.0,
    "wheel_slip_front_right": 25.0,
    "wheel_slip_rear_left": 25.0,
    "wheel_slip_rear_right": 25.0,
}


class EnvironmentState:
    def __init__(self, config: Dict):
        self._setup(config)

    def reset(self):
        self._reset_history()

    def step(self, observation: Dict) -> np.array:
        observation = observation["state"]
        self._add_calculated_values(observation)
        self._current_observation = observation
        current_state = self._state_representation_from_observation(observation)
        past_state = self._historical_state()
        self._update_history(current_state)
        expanded_state = self._add_additional_state(current_state)
        return np.hstack([expanded_state, past_state])

    def _add_calculated_values(self, observation: Dict):
        curvature_ahead, distance_to_raceline = self._look_ahead(observation)
        observation["gap"] = distance_to_raceline
        observation["curvature"] = curvature_ahead

    def _state_representation_from_observation(self, observation: Dict) -> np.array:
        state = [observation[key] for key in STATE_REPRESENTATION]
        return np.array(state, dtype=np.float32) / self._normalisation

    def _historical_state(self) -> np.array:
        return np.hstack([state for state in self._history])

    def _update_history(self, latest_state: np.array):
        if self._is_full:
            self._history.popleft()
            self._history.append(latest_state)
        else:
            self._history.append(latest_state)

    @property
    def _is_full(self) -> bool:
        return len(self._history) == self._history.maxlen

    def _add_additional_state(self, state: np.array) -> np.array:
        additional_state = np.zeros(self._additional_state_dimension, dtype=np.float32)
        additional_state[0] = self._is_outside_tracklimits
        n_points = self._n_curvature_points
        curvature = self._current_observation["curvature"] / CURVATURE_NORMALISATION
        additional_state[1 : n_points + 1] = curvature
        return np.hstack([state, additional_state])

    @property
    def _is_outside_tracklimits(self) -> bool:
        return self._current_observation["number_of_tyres_out"] > 2

    def __getitem__(self, key: str) -> Union[int, float, str]:
        return self._current_observation[key]

    def _setup(self, config: Dict):
        self._unpack_config(config)
        self._setup_look_ahead_curve()
        self._setup_state_dimension()
        self._setup_normalising_factors()
        self._reset_history()

    def _unpack_config(self, config: Dict):
        self._config = config
        self._n_steps = config["n_steps"]

    def _setup_look_ahead_curve(self):
        self._look_ahead = LookAhead(self._config["look_ahead"])
        self._n_curvature_points = self._config["look_ahead"]["n_points"]

    def _setup_state_dimension(self):
        single_representation = len(STATE_REPRESENTATION)
        state_dimension = single_representation * (self._n_steps + 1)
        self._additional_state_dimension = 1 + self._n_curvature_points
        self.state_dimension = state_dimension + self._additional_state_dimension

    def _setup_normalising_factors(self):
        factors = [STATE_REPRESENTATION[key] for key in STATE_REPRESENTATION]
        self._normalisation = np.array(factors, dtype=np.float32)

    def _reset_history(self):
        self._history = deque(maxlen=self._n_steps)
        self._fill_history()

    def _fill_history(self):
        for _ in range(self._n_steps):
            self._update_history(np.zeros(len(STATE_REPRESENTATION)))
