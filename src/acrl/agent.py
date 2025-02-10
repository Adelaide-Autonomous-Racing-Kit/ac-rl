from collections import namedtuple
import time
from typing import Dict

from aci.interface import AssettoCorsaInterface
from acrl.buffer.replay_buffer import ReplayBuffer
from acrl.buffer.utils import BehaviouralSample
from acrl.sac.sac import SoftActorCritic
from acrl.utils import load
from acrl.utils.state import EnvironmentState
import numpy as np
import wandb

Limit = namedtuple("Limits", "min max rate")
ControlLimits = namedtuple("ControlLimits", "steer pedal")

SAMPLING_FREQUENCY = 25  # Hz

STEERING_LIMIT = Limit(-1.0, 1.0, 600 / 180 * 1 / SAMPLING_FREQUENCY)
PEDAL_LIMIT = Limit(
    0.0, 1.0, 1200 / 100 * 1 / SAMPLING_FREQUENCY
)  # Rate * control frequency
CONTROL_LIMITS = ControlLimits(STEERING_LIMIT, PEDAL_LIMIT)

MINIMUM_SPEED_KMH = 3.6
RESTART_PATIENCE = 10 * SAMPLING_FREQUENCY


class SACAgent(AssettoCorsaInterface):
    def __init__(self, config_path: str):
        self.cfg = load.yaml(config_path)
        super().__init__(self.cfg["aci"])
        self.setup()

    def behaviour(self, observation: Dict) -> np.array:
        start_time = time.time_ns()
        representation = self._environment_state.step(observation)
        self._update_buffer(representation)

        action = self._get_action(representation)
        self._update_control(action)
        self.act(self._current_action)

        self._maybe_update_policy()

        self._previous_action = action
        self._previous_representation = representation

        while (time.time_ns() - start_time) < 40_000_000:
            # 25Hz rate limit
            continue

    def _update_control(self, action: np.array) -> np.array:
        steer_rate = CONTROL_LIMITS.steer.rate
        pedal_rate = CONTROL_LIMITS.pedal.rate
        rates = np.array([steer_rate, pedal_rate, pedal_rate])
        deltas = action * rates
        self._current_action += deltas
        steer_min, steer_max = CONTROL_LIMITS.steer.min, CONTROL_LIMITS.steer.max
        pedal_min, pedal_max = CONTROL_LIMITS.pedal.min, CONTROL_LIMITS.pedal.max
        minimums = np.array([steer_min, pedal_min, pedal_min])
        maximums = np.array([steer_max, pedal_max, pedal_max])
        np.clip(self._current_action, minimums, maximums, self._current_action)

    def _update_buffer(self, representation: np.array):
        reward = self._reward()
        if self._previous_representation is not None:
            sample = BehaviouralSample(
                action=self._previous_action,
                done=self._is_done,
                reward=reward,
                next_state=representation,
                state=self._previous_representation,
            )
            self._replay_buffer.append(sample)
        self._episode_reward += reward

    def _reward(self) -> float:
        state = self._environment_state
        reward = float(state["speed_kmh"])  # * ( 1.0 - (np.abs( state["gap"]) / 12.00))
        reward /= 300.0  # normalize
        return reward

    def _get_action(self, representation: np.array) -> np.array:
        # Fills the n step return buffer on restart
        if self._n_actions < self._n_step_buffer_states:
            action = self._default_action
        # Generates random actions to warm up training examples
        if self._start_steps > self._n_actions:
            action = self._random_action()
        else:
            action, _ = self._sac.explore(representation)
        self._n_actions += 1
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
                train_stats = self._sac.update_online_networks(batch)
                if train_stats:
                    print(train_stats)

            # Update target networks.
            self._sac.update_target_networks()

    def teardown(self):
        self._wandb_run.finish()

    def termination_condition(self, observation: Dict) -> bool:
        return False

    def restart_condition(self, observation: Dict) -> bool:
        is_done = False
        is_done = is_done or self._is_outside_track_limits(observation)
        # is_done = is_done or self._is_progressing_too_slowly(observation)
        # is_done = is_done or self._is_agent_off_raceline(observation)
        self._is_done = is_done
        return is_done

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

    def _is_agent_off_raceline(self, observation: Dict) -> bool:
        is_done = False
        return is_done

    def on_restart(self):
        wandb.log({"reward/episode_reward": self._episode_reward}, self._n_episodes)
        self._reset_episode()
        self._n_episodes += 1

    def setup(self):
        self._setup_wandb()
        self._reset_episode()
        input_dim = self._environment_state.state_dimension
        self.cfg["sac"]["policy"]["input_dim"] = input_dim
        self._sac = SoftActorCritic(self.cfg["sac"])
        self._n_step_buffer_states = self.cfg["sac"]["n_steps"]
        self._replay_buffer = ReplayBuffer(self.cfg)
        self._start_steps = self.cfg["training"]["start_steps"]
        self._update_interval = self.cfg["training"]["update_interval"]
        self._batch_size = self.cfg["training"]["batch_size"]
        self._default_action = np.array([0.0, -1.0, -1.0])
        self._n_actions = 0
        self._n_episodes = 0

    def _setup_wandb(self):
        config = self.cfg["wandb"]
        self._wandb_run = wandb.init(
            entity=config["entity"],
            project=config["project_name"],
            name=config["run_name"],
        )

    def _reset_episode(self):
        self._is_done = False
        self._episode_reward = 0
        self._previous_action = None
        self._previous_representation = None
        self._minimum_speed_patience = RESTART_PATIENCE
        self._current_action = np.array([0.0, -1.0, -1.0])
        self._environment_state = EnvironmentState(self.cfg["sac"])
