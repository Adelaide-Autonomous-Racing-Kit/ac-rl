from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class NStepBehaviouralSample:
    action: np.array
    reward: float
    state: np.array

    def unpack(self) -> Tuple[np.array, float, np.array]:
        return self.action, self.reward, self.state


@dataclass
class BehaviouralSample:
    action: np.array
    done: bool
    reward: float
    next_state: np.array
    state: np.array

    def unpack(self) -> Tuple[np.array, float, np.array]:
        return self.action, self.reward, self.state

    def update(self, action: np.array, reward: float, state: np.array):
        self.action, self.reward, self.state = action, reward, state

    @property
    def is_episode_finished(self) -> bool:
        return self.done
