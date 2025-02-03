import numpy as np

from aci.interface import AssettoCorsaInterface
from acrl.sac.sac import SoftActorCritic
from acrl.buffer.replay_buffer import ReplayBuffer
from acrl.buffer.utils import BehaviouralSample
from acrl.utils import load


class SACAgent(AssettoCorsaInterface):
    def __init__(self, config_path: str):
        self.cfg = load.yaml(config_path)
        super().__init__(self.cfg["aci"])
        self.setup()

    def behaviour(self, observation: Dict) -> np.array:
        self._update_buffer(observation)
        action = self._get_action(observation)
        self._previous_action = action
        self._previous_state = observation
        return action

    def _update_buffer(self, observation: Dict):
        # TODO: Move this to another process
        sample = BehaviouralSample(
            action=self._previous_action,
            done=,
            reward=self._reward(self._previous_state),
            next_state=observation,
            state=self._previous_state,
            terminated=,
        )
        self._replay_buffer.append(sample)
    
    def _reward(self, state) -> float:
        return 1.0

    def _get_action(self, observation: Dict) -> np.array:
        if self._start_steps > self._steps:
            action = self._action_space.sample()
        else:
            action, _ = self._sac.explore(state)
        return action

    def teardown(self):
        pass

    def termination_condition(self, observation: Dict) -> bool:
        return False

    def restart_condition(self, observation: Dict) -> bool:
        return False

    def setup(self):
        self._sac = SoftActorCritic(self.cfg["sac"])
        self._replay_buffer = ReplayBuffer(self.cfg)
        self._previous_action = np.zeros(3)
        self._previous_state = np.zeros(100)
