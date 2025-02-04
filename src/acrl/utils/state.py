from typing import Dict, Union

import numpy as np


AGENT_RECEIVED_STATE = [
    'speed_kmh',
    'gap',
    'final_force_feedback',
    'rpm',
    'acceleration_g_X',
    'acceleration_g_Y',
    'gear',
    'local_angular_velocity_Y',
    'local_velocity_X',
    'local_velocity_Y',
    'wheel_slip_front_left',
    'wheel_slip_front_right',
    'wheel_slip_rear_left',
    'wheel_slip_rear_right',
]


class EnvironmentState:
    def __init__(self, observation: Dict):
        # TODO: Implement this
        observation["state"]["gap"] = 0.0
        self._observation = observation['state']
        self._setup_agent_state()
    
    def _setup_agent_state(self):
        self._state_dimension = len(AGENT_RECEIVED_STATE)
        state = [ self._observation[key] for key in AGENT_RECEIVED_STATE ]
        self._agent_state = np.array(state, dtype=np.float32)

    @property
    def representation(self) -> np.array:
        return self._agent_state

    def __getitem__(self, key:str) -> Union[int, float, str]:
        return self._observation[key]
    