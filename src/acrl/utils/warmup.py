from typing import Dict

import numpy as np


class SpeedRewardWarmUp:
    def __init__(self, config: Dict):
        self.__setup(config)

    def __call__(self, speed: float) -> float:
        return np.clip(speed, a_min=0.0, a_max=self.reward_limit)

    @property
    def reward_limit(self) -> float:
        if self._n_truncated_episodes <= self._n_eps_before_warmup:
            return self._initial_reward_limit
        elif self._n_episodes_into_warmup <= self._n_warmup_eps:
            return self._initial_reward_limit + self._limit_increase
        return self._max_reward_limit

    @property
    def _limit_increase(self) -> float:
        return self._increase * self._n_episodes_into_warmup

    @property
    def _n_episodes_into_warmup(self) -> float:
        return self._n_truncated_episodes - self._n_eps_before_warmup

    def increment_truncated_episodes(self):
        self._n_truncated_episodes += 1

    def __setup(self, config: Dict):
        self._initial_reward_limit = config["max_speed_reward"]["initial"]
        self._max_reward_limit = config["max_speed_reward"]["final"]
        self._n_eps_before_warmup = config["n_truncated_eps"]
        self._n_warmup_eps = config["n_warmup_eps"]
        speed_difference = self._max_reward_limit - self._initial_reward_limit
        self._increase = speed_difference / self._n_warmup_eps
        self._n_truncated_episodes = 0
