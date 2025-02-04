from typing import Dict, NamedTuple, Tuple

import torch
import numpy as np
from acrl.buffer.n_step_buffer import NStepBuffer
from acrl.buffer.utils import BehaviouralSample


class SampleBatch(NamedTuple):
    actions: torch.Tensor
    dones: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    states: torch.Tensor

    def unpack(self)-> Tuple[torch.Tensor,...]:
        return self.actions, self.dones, self.next_states, self.rewards, self.states


class ReplayBuffer:
    def __init__(self, config: Dict):
        self._setup(config)

    def reset(self):
        self._allocate_buffers()
        self._n_buffered = 0

    def _allocate_buffers(self):
        self._maybe_allocate_n_step_buffer()
        self._states = self._allocate_state_buffer()
        self._next_states = self._allocate_state_buffer()
        self._actions = self._allocate_action_buffer()
        self._rewards = self._allocate_scalar_buffer()
        self._dones = self._allocate_scalar_buffer()

    def _maybe_allocate_n_step_buffer(self):
        if self._is_using_n_step_rewards:
            self._n_step_buffer = NStepBuffer(self._discount, self._n_steps)

    def _allocate_state_buffer(self) -> np.array:
        return self._allocate_empty_array(self._state_buffer_shape)

    def _allocate_action_buffer(self) -> np.array:
        return self._allocate_empty_array(self._action_buffer_shape)

    def _allocate_scalar_buffer(self) -> np.array:
        return self._allocate_empty_array((self._max_buffered_samples, 1))

    @property
    def _state_buffer_shape(self) -> Tuple[int, int]:
        return (self._max_buffered_samples, self._state_shape)

    @property
    def _action_buffer_shape(self) -> Tuple[int, int]:
        return (self._max_buffered_samples, self._action_shape)

    def _allocate_empty_array(self, size: Tuple[int, ...]) -> np.array:
        return np.empty(size, dtype=np.float32)

    def append(self, sample: BehaviouralSample):
        if self._is_using_n_step_rewards:
            self._append_n_step_reward(sample)
        else:
            self._append(sample)

    @property
    def _is_using_n_step_rewards(self) -> bool:
        return self._n_steps > 1

    def _append_n_step_reward(self, sample: BehaviouralSample):
        self._n_step_buffer.append(sample)
        if self._n_step_buffer.is_full():
            self._add_step_to_buffer(sample)
        if sample.is_episode_finished:
            self._add_remaining_steps_to_buffer(sample)

    def _add_step_to_buffer(self, sample: BehaviouralSample):
        self._update_sample_with_n_step_reward(sample)
        self._append(sample)

    def _update_sample_with_n_step_reward(self, sample: BehaviouralSample):
        n_step_sample = self._n_step_buffer.popleft()
        action, reward, state = n_step_sample.unpack()
        sample.update(action, reward, state)

    def _add_remaining_steps_to_buffer(self, sample: BehaviouralSample):
        while not self._n_step_buffer.is_empty():
            self._add_step_to_buffer(sample)

    def _append(self, sample: BehaviouralSample):
        self._states[self._buffer_index, ...] = sample.state
        self._actions[self._buffer_index, ...] = sample.action
        self._rewards[self._buffer_index, ...] = sample.reward
        self._next_states[self._buffer_index, ...] = sample.next_state
        self._dones[self._buffer_index, ...] = sample.done
        self._n_buffered += 1

    @property
    def _buffer_index(self) -> int:
        return self._n_buffered % self._max_buffered_samples

    def sample(self, batch_size: int) -> torch.Tensor:
        indices = self._sample_indices(batch_size)
        return self._sample_batch(indices)

    def _sample_indices(self, batch_size: int) -> np.array:
        return np.random.randint(low=0, high=self.n_buffered, size=batch_size)

    def _sample_batch(self, indices: np.array) -> SampleBatch:
        states = self._sample_array(self._states, indices)
        actions = self._sample_array(self._actions, indices)
        rewards = self._sample_array(self._rewards, indices)
        dones = self._sample_array(self._dones, indices)
        next_states = self._sample_array(self._next_states, indices)
        return SampleBatch(actions, dones, next_states, rewards, states)

    def _sample_array(self, array: np.array, indices: np.array) -> torch.Tensor:
        return torch.tensor(array[indices], dtype=torch.float, device=self._device)

    def __len__(self):
        return self.n_buffered

    @property
    def n_buffered(self) -> int:
        return min(self._n_buffered, self._max_buffered_samples)

    def _setup(self, config: Dict):
        self._unpack_config(config)
        self._setup_accelerator()
        self.reset()

    def _unpack_config(self, config: Dict):
        self._max_buffered_samples = config["training"]["buffer_size"]
        self._state_shape = config["sac"]["policy"]["input_dim"]
        self._action_shape = config["sac"]["policy"]["output_dim"]
        self._discount = config["sac"]["gamma"]
        self._n_steps = config["sac"]["n_steps"]

    def _setup_accelerator(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
