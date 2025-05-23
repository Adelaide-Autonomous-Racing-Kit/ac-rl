from pathlib import Path
import random
import time
from typing import Dict

from aci.interface import AssettoCorsaInterface
from acrl.buffer.replay_buffer import ReplayBuffer
from acrl.buffer.utils import BehaviouralSample
from acrl.sac.sac import SoftActorCritic
from acrl.utils import load
from acrl.utils.checkpointer import Checkpointer
from acrl.utils.constants import (
    CONTROL_MAXS,
    CONTROL_MINS,
    CONTROL_RATES,
    MAX_EPISODE_LENGTH,
    MINIMUM_SPEED_KMH,
    RESTART_PATIENCE,
    SAMPLING_FREQUENCY,
)
from acrl.utils.state import EnvironmentState
from acrl.utils.warmup import SpeedRewardWarmUp
import numpy as np
import torch
import wandb

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)


class SACAgent(AssettoCorsaInterface):
    def __init__(self, config_path: str):
        self.cfg = load.yaml(config_path)
        super().__init__(self.cfg["aci"])
        self.setup()

    def behaviour(self, observation: Dict) -> np.array:
        start_time = time.time()
        representation = self._environment_state.step(observation)

        if self._is_training:
            self._training_behaviour(representation)
        else:
            self._evaluation_behaviour(representation)

        self._rate_limit(start_time)

    def _training_behaviour(self, representation: np.array):
        self._update_buffer(representation)

        action = self._get_training_action(representation)
        self._update_control(action)
        self.act(self._current_action)

        self._maybe_update_policy()

        self._previous_action = action
        self._previous_representation = representation

    def _update_control(self, action: np.array) -> np.array:
        deltas = action * CONTROL_RATES
        self._current_action += deltas
        np.clip(self._current_action, CONTROL_MINS, CONTROL_MAXS, self._current_action)

    def _update_buffer(self, representation: np.array):
        reward = self._reward()
        if self._previous_representation is not None:
            sample = BehaviouralSample(
                action=self._previous_action,
                done=self._is_done,
                reward=reward,
                next_state=representation,
                state=self._previous_representation,
                truncated=self._is_truncated,
            )
            self._replay_buffer.append(sample)
        self._episode_reward += reward

    def _reward(self) -> float:
        state = self._environment_state
        speed = self._reward_warmup(state["speed_kmh"])
        reward = speed * (1.0 - (np.abs(state["gap"]) / 12.00))
        reward /= self._maximum_speed
        return reward

    def _get_training_action(self, representation: np.array) -> np.array:
        # Fills the n step return buffer on restart
        if self._n_actions < self._n_step_buffer_states:
            action = self._default_action
        # Generates random actions to warm up training examples
        if self._start_steps > self._n_actions:
            action = self._random_action()
        else:
            action, _ = self._sac.explore(representation)
        self._n_actions += 1
        self._episode_length += 1
        return action

    def _random_action(self) -> np.array:
        action = np.random.rand(3)
        # Rescale to be between [-1., 1]
        action = (action - 0.5) * 2
        return action

    def _maybe_update_policy(self):
        if self._n_actions > self._start_steps:
            if self._n_actions % self._update_interval == 0:
                batch = self._replay_buffer.sample(self._batch_size)
                self._sac.update_online_networks(batch)
            self._sac.update_target_networks()

    def _evaluation_behaviour(self, representation: np.array):
        action = self._get_evaluation_action(representation)
        self._update_control(action)
        self.act(self._current_action)

    def _get_evaluation_action(self, representation: np.array) -> np.array:
        actions, _ = self._sac.exploit(representation)
        return actions

    def _rate_limit(self, start_time: float):
        while (time.time() - start_time) < (1 / SAMPLING_FREQUENCY):
            # Rate limit to sampling frequency
            continue

    def teardown(self):
        self._checkpoint()
        self._wandb_run.finish()

    def termination_condition(self, observation: Dict) -> bool:
        return False

    def restart_condition(self, observation: Dict) -> bool:
        # Negative Episode Termination Conditions
        is_done = False
        is_done = is_done or self._is_outside_track_limits(observation)
        is_done = is_done or self._is_progressing_too_slowly(observation)
        is_done = is_done or self._is_too_far_away_from_raceline()
        self._is_done = is_done
        # Positive Episode Termination Conditions
        is_truncated = False
        is_truncated = is_truncated or self._is_episode_too_long()
        self._is_truncated = is_truncated
        # Restart flag
        return is_done or is_truncated

    def _is_outside_track_limits(self, observation: Dict) -> bool:
        return observation["state"]["number_of_tyres_out"] > 2

    def _is_progressing_too_slowly(self, observation: Dict) -> bool:
        is_done = False
        if observation["state"]["speed_kmh"] < MINIMUM_SPEED_KMH:
            self._minimum_speed_patience -= 1
        else:
            self._minimum_speed_patience = RESTART_PATIENCE
        if self._minimum_speed_patience < 1:
            is_done = True
        return is_done

    def _is_too_far_away_from_raceline(self):
        return abs(self._environment_state["gap"]) > 12.0

    def _is_episode_too_long(self) -> bool:
        return self._episode_length > MAX_EPISODE_LENGTH

    def on_restart(self):
        wandb.log({"policy/reward": self._episode_reward})
        self._maybe_checkpoint_training()
        self._update_reward_warmup()
        self._update_training_flag()
        self._reset_episode()

    def _maybe_checkpoint_training(self):
        actions_since_checkpoint = self._n_actions - self._last_checkpoint
        if actions_since_checkpoint >= self._checkpoint_interval:
            self._checkpoint()

    def _checkpoint(self):
        self._checkpointer.checkpoint(self._n_actions)
        self._last_checkpoint = self._n_actions

    def _update_reward_warmup(self):
        if self._is_truncated:
            self._reward_warmup.increment_truncated_episodes()
        wandb.log({"policy/max_speed_reward": self._reward_warmup.reward_limit})

    def _update_training_flag(self):
        # TODO: Implement evaluation
        self._is_training = True

    def setup(self):
        self._unpack_config()
        self._setup_wandb()
        self._setup_environment()
        self._setup_SAC()
        self._setup_replay_buffer()
        self._setup_defaults()
        self._setup_checkpointer()
        self._setup_reward_warmup()

    def _unpack_config(self):
        self._n_step_buffer_states = self.cfg["sac"]["n_steps"]
        self._start_steps = self.cfg["training"]["start_steps"]
        self._update_interval = self.cfg["training"]["update_interval"]
        self._batch_size = self.cfg["training"]["batch_size"]
        run_name = self.cfg["wandb"]["run_name"]
        checkpoint_path = self.cfg["training"]["checkpoint"]["path"]
        self._checkpoint_path = Path(f"{checkpoint_path}/{run_name}")
        self._checkpoint_interval = self.cfg["training"]["checkpoint"]["interval"]

    def _setup_wandb(self):
        config = self.cfg["wandb"]
        self._wandb_run = wandb.init(
            entity=config["entity"],
            project=config["project_name"],
            name=config["run_name"],
        )

    def _setup_environment(self):
        self._environment_state = EnvironmentState(self.cfg["sac"])
        input_dim = self._environment_state.state_dimension
        self.cfg["sac"]["policy"]["input_dim"] = input_dim

    def _setup_SAC(self):
        self._sac = SoftActorCritic(self.cfg["sac"])

    def _setup_replay_buffer(self):
        self._replay_buffer = ReplayBuffer(self.cfg)

    def _setup_checkpointer(self):
        self._last_checkpoint = 0
        path = self._checkpoint_path
        config = self.cfg["training"]["checkpoint"]
        self._checkpointer = Checkpointer(config, path, self._sac, self._replay_buffer)
        if config["resume"]:
            self._n_actions = self._checkpointer.load(config["name"])
            self._last_checkpoint = self._n_actions

    def _setup_defaults(self):
        self._default_action = np.array([0.0, -1.0, -1.0])
        self._n_actions = 0
        self._is_training = True
        self._reset_episode()

    def _reset_episode(self):
        self._is_done = False
        self._is_truncated = False
        self._episode_reward = 0
        self._episode_length = 0
        self._previous_action = None
        self._previous_representation = None
        self._minimum_speed_patience = RESTART_PATIENCE
        self._current_action = np.array([0.0, -1.0, -1.0])
        self._environment_state.reset()

    def _setup_reward_warmup(self):
        config = self.cfg["training"]["speed_reward_warmup"]
        self._reward_warmup = SpeedRewardWarmUp(config)
        self._maximum_speed = config["max_speed_reward"]["final"]
