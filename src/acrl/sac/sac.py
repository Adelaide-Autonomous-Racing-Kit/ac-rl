from typing import Dict, Tuple, Union

import torch
import numpy as np
from torch.optim import Adam

from acrl.sac.modelling.policies import GaussianActionPolicy, TwinActionPolicy
from acrl.buffer.replay_buffer import SampleBatch


class SoftActorCritic:
    def __init__(self, config: Dict):
        self._setup(config)

    def explore(self, state: np.array) -> Tuple[np.array, torch.Tensor]:
        state = self._to_tensor(state)
        with torch.no_grad():
            action, entropies, _ = self._policy(state)
        action = action.cpu().numpy()[0]
        return action, entropies

    def _to_tensor(self, state: np.array) -> torch.Tensor:
        state = state[None, ...].copy()
        return torch.tensor(state, dtype=torch.float, device=self._device)

    def exploit(self, state: np.array) -> Tuple[np.array, torch.Tensor]:
        state = self._to_tensor(state)
        with torch.no_grad():
            _, entropies, action = self._policy(state)
        action = action.cpu().numpy()[0]
        return action, entropies

    def update_target_networks(self):
        for t, s in zip(self._target_q.parameters(), self._online_q.parameters()):
            self._soft_update(t, s)

    def _soft_update(self, target: torch.Tensor, source: torch.Tensor):
        tau = self._target_update_coef
        target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def update_online_networks(self, batch: SampleBatch):
        self._n_learning_steps += 1
        stats = self.update_policy_and_entropy(batch)
        self.update_q_functions(batch)
        return stats

    def update_policy_and_entropy(
        self,
        batch: SampleBatch,
    ) -> Union[Dict, None]:
        policy_loss, entropies = self._policy_loss(batch.states)
        self._step_policy_optimiser(policy_loss)
        entropy_loss = self._entropy_loss(entropies)
        self._step_alpha_optimiser(entropy_loss)
        self._alpha = self._log_alpha.detach().exp()
        return self._maybe_log_policy_update(
            policy_loss, entropy_loss, entropies
        )

    def _step_policy_optimiser(self, loss: torch.Tensor):
        self._step_optimiser(self._policy_optim, loss)

    def _step_alpha_optimiser(self, loss: torch.Tensor):
        self._step_optimiser(self._alpha_optim, loss)

    def _step_q_optimiser(self, loss: torch.Tensor):
        self._step_optimiser(self._q_optim, loss)

    def _step_optimiser(self, optimiser: torch.optim.Optimizer, loss: torch.Tensor):
        optimiser.zero_grad()
        loss.backward(retain_graph=False)
        optimiser.step()

    def _policy_loss(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self._policy(states)
        # Expectations of Q with clipped double Q technique.
        qs1, qs2 = self._online_q(states, sampled_actions)
        qs = torch.min(qs1, qs2)
        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((-qs - self._alpha * entropies))
        return policy_loss, entropies.detach_()

    def _entropy_loss(self, entropies: torch.Tensor) -> torch.Tensor:
        # Increase alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -torch.mean(self._log_alpha * (self._target_entropy - entropies))
        return entropy_loss

    def _maybe_log_policy_update(
        self,
        policy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        entropies: torch.Tensor,
    ) -> Union[Dict, None]:
        if self._is_logging_step():
            return self._log_policy_update(policy_loss, entropy_loss, entropies)

    def _is_logging_step(self) -> bool:
        return self._n_learning_steps % self._log_interval == 0

    def _log_policy_update(
        self,
        policy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        entropies: torch.Tensor,
    ) -> Dict:
        policy_loss = policy_loss.detach().item()
        entropy_loss = entropy_loss.detach().item()
        entropies = entropies.detach().mean().item()
        # writer.add_scalar("loss/policy", policy_loss, self._n_learning_steps)
        # writer.add_scalar("loss/entropy", entropy_loss, self._n_learning_steps)
        # writer.add_scalar("stats/alpha", self._alpha.item(), self._n_learning_steps)
        # writer.add_scalar("stats/entropy", entropies, self._n_learning_steps)
        to_log = {
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "alpha": self._alpha.item(),
            "entropy": entropies,
        }
        return to_log

    def update_q_functions(
        self,
        batch: SampleBatch,
        q1_loss_weights=None,
        q2_loss_weights=None,
    ):
        actions, dones, next_states, rewards, states = batch.unpack()
        current_q1, current_q2 = self._online_q(states, actions)
        target_qs = self._target_qs(rewards, next_states, dones)
        q_loss = self._q_loss(
            current_q1,
            current_q2,
            target_qs,
            q1_loss_weights,
            q2_loss_weights,
        )
        self._step_q_optimiser(q_loss)
        self._maybe_log_q_update(q_loss, current_q1, current_q2)

    def _maybe_log_q_update(
        self,
        q_loss: torch.Tensor,
        current_q1: torch.Tensor,
        current_q2: torch.Tensor,
    ):
        if self._is_logging_step():
            self._log_q_update(q_loss, current_q1, current_q2)

    def _log_q_update(
        self,
        q_loss: torch.Tensor,
        current_q1: torch.Tensor,
        current_q2: torch.Tensor,
    ):
        q1_mean = current_q1.detach().mean().item()
        q2_mean = current_q2.detach().mean().item()
        # writer.add_scalar("loss/Q", q_loss.detach().item(), self._n_learning_steps)
        # writer.add_scalar("stats/mean_Q1", q1_mean, self._n_learning_steps)
        # writer.add_scalar("stats/mean_Q2", q2_mean, self._n_learning_steps)

    def _target_qs(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy(next_states)
            next_qs1, next_qs2 = self._target_q(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) + self._alpha * next_entropies
        target_qs = rewards + (1.0 - dones) * self._discount * next_qs
        return target_qs

    def _q_loss(
        self,
        current_q1: torch.Tensor,
        current_q2: torch.Tensor,
        target_qs: torch.Tensor,
        q1_loss_weights: torch.Tensor = None,
        q2_loss_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        # Q loss is mean squared TD errors with importance weights.
        if q1_loss_weights is None:
            q1_loss = torch.mean((current_q1 - target_qs).pow(2))
            q2_loss = torch.mean((current_q2 - target_qs).pow(2))
        else:
            q1_loss = torch.sum((current_q1 - target_qs).pow(2) * q1_loss_weights)
            q2_loss = torch.sum((current_q2 - target_qs).pow(2) * q2_loss_weights)
        return q1_loss + q2_loss

    def _setup(self, config: Dict):
        self._unpack_config(config)
        self._setup_accelerator()
        self._setup_counters()
        self._setup_policies()

    def _unpack_config(self, config: Dict):
        self._model_config = config["policy"]
        self._gamma = config["gamma"]
        self._n_steps = config["n_steps"]
        self._discount = self._gamma**self._n_steps
        self._log_interval = config["log_interval"]
        self._policy_lr = config["policy_lr"]
        self._entropy_lr = config["entropy_lr"]
        self._q_lr = config["q_lr"]
        self._target_update_coef = config["target_update_coef"]

    def _setup_accelerator(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_counters(self):
        self._n_learning_steps = 0

    def _setup_policies(self):
        self._policy = GaussianActionPolicy(self._model_config)
        self._policy.to(self._device)
        self._online_q = TwinActionPolicy(self._model_config)
        self._online_q.to(self._device)
        self._target_q = TwinActionPolicy(self._model_config)
        self._target_q.load_state_dict(self._online_q.state_dict())
        self._target_q.eval()
        self._target_q.to(self._device)
        self._policy_optim = Adam(self._policy.parameters(), lr=self._policy_lr)
        self._q_optim = Adam(self._online_q.parameters(), lr=self._q_lr)
        self._target_entropy = -float(self._model_config["output_dim"])
        self._log_alpha = torch.zeros(1, device=self._device, requires_grad=True)
        self._alpha = self._log_alpha.detach().exp()
        self._alpha_optim = Adam([self._log_alpha], lr=self._entropy_lr)
