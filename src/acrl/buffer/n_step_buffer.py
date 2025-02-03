from collections import deque

import numpy as np
from acrl.buffer.utils import BehaviouralSample, NStepBehaviouralSample


class NStepBuffer:
    """
    Provides rewards for agent actions based on the next n actions
    """

    def __init__(self, discount: float, n_steps: int):
        self._n_steps = n_steps
        self._discounts = np.array([discount**i for i in range(n_steps)])
        self.reset()

    def reset(self):
        self._samples = deque(maxlen=self._n_steps)

    def append(self, sample: BehaviouralSample):
        self._samples.append(self._unpack_sample(sample))

    def _unpack_sample(self, sample: BehaviouralSample) -> NStepBehaviouralSample:
        action, reward, state = sample.unpack()
        return NStepBehaviouralSample(action, reward, state)

    def is_empty(self) -> bool:
        return len(self._samples) == 0

    def is_full(self) -> bool:
        return len(self._samples) == self._samples.maxlen

    def popleft(self) -> NStepBehaviouralSample:
        n_step_reward = self._n_step_reward()
        sample = self._samples.popleft()
        sample.reward = n_step_reward
        return sample

    def _n_step_reward(self) -> float:
        return np.dot(self._rewards, self._discounts)

    @property
    def _rewards(self) -> np.array:
        return np.array([sample.reward for sample in self._samples])

    def __len__(self) -> int:
        return len(self._samples)
