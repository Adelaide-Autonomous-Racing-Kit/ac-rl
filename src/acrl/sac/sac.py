from typing import Dict, Tuple

from acrl.buffer.replay_buffer import SampleBatch
from acrl.sac.modelling.policies import GaussianActionPolicy, TwinQNetwork
from loguru import logger
import numpy as np
import torch
from torch.optim import Adam
import wandb


class SoftActorCritic:
    def __init__(self, config: Dict):
        self._setup(config)

    def explore(self, state: np.array) -> Tuple[np.array, torch.Tensor]:
        state = self._to_tensor(state)
        with torch.no_grad():
            action, entropies, _ = self.policy(state)
        action = action.cpu().numpy()[0]
        return action, entropies

    def _to_tensor(self, state: np.array) -> torch.Tensor:
        state = state[None, ...].copy()
        return torch.tensor(state, dtype=torch.float, device=self._device)

    def exploit(self, state: np.array) -> Tuple[np.array, torch.Tensor]:
        state = self._to_tensor(state)
        with torch.no_grad():
            _, entropies, action = self.policy(state)
        action = action.cpu().numpy()[0]
        return action, entropies

    def update_target_networks(self):
        for t, s in zip(self.target_q.parameters(), self.online_q.parameters()):
            self._soft_update(t, s)

    def _soft_update(self, target: torch.Tensor, source: torch.Tensor):
        tau = self._target_update_coef
        target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def update_online_networks(self, batch: SampleBatch):
        self._n_learning_steps += 1
        stats = self.update_policy_and_entropy(batch)
        self.update_q_functions(batch)
        return stats

    def update_policy_and_entropy(self, batch: SampleBatch):
        policy_loss, entropies = self._policy_loss(batch.states)
        self._step_policy_optimiser(policy_loss)
        entropy_loss = self._entropy_loss(entropies)
        self._step_alpha_optimiser(entropy_loss)
        self._alpha = self.log_alpha.detach().exp()
        self._maybe_log_policy_update(policy_loss, entropy_loss, entropies)

    def _step_policy_optimiser(self, loss: torch.Tensor):
        self._step_optimiser(self.policy_optim, loss)

    def _step_alpha_optimiser(self, loss: torch.Tensor):
        self._step_optimiser(self.alpha_optim, loss)

    def _step_q_optimiser(self, loss: torch.Tensor):
        self._step_optimiser(self.q_optim, loss)

    def _step_optimiser(self, optimiser: torch.optim.Optimizer, loss: torch.Tensor):
        optimiser.zero_grad()
        loss.backward(retain_graph=False)
        optimiser.step()

    def _policy_loss(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self.policy(states)
        # Expectations of Q with clipped double Q technique.
        qs1, qs2 = self.online_q(sampled_actions, states)
        qs = torch.min(qs1, qs2)
        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((-qs - self._alpha * entropies))
        return policy_loss, entropies.detach_()

    def _entropy_loss(self, entropies: torch.Tensor) -> torch.Tensor:
        # Increase alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -torch.mean(self.log_alpha * (self._target_entropy - entropies))
        return entropy_loss

    def _maybe_log_policy_update(
        self,
        policy_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        entropies: torch.Tensor,
    ):
        if self._is_logging_step():
            self._log_policy_update(policy_loss, entropy_loss, entropies)

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
        to_log = {
            "policy/policy_loss": policy_loss,
            "policy/entropy_loss": entropy_loss,
            "policy/alpha": self._alpha.item(),
            "policy/entropy_mean": entropies,
        }
        wandb.log(to_log, self._n_learning_steps)
        self._log_to_console(to_log)

    def update_q_functions(
        self,
        batch: SampleBatch,
        q1_loss_weights=None,
        q2_loss_weights=None,
    ):
        current_q1, current_q2 = self.online_q(batch.actions, batch.states)
        target_qs = self._target_qs(batch.rewards, batch.next_states, batch.dones)
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
        to_log = {
            "Q/Q_loss": q_loss.detach().item(),
            "Q/Q1_mean": q1_mean,
            "Q/Q2_mean": q2_mean,
        }
        wandb.log(to_log, self._n_learning_steps)
        self._log_to_console(to_log)

    def _log_to_console(self, to_log: Dict):
        message = f"{self._n_learning_steps}: "
        for key in to_log:
            message += f"{key} - {to_log[key]}, "
        logger.info(message[:-2])

    def _target_qs(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy(next_states)
            next_qs1, next_qs2 = self.target_q(next_actions, next_states)
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

    def serialise(self) -> Dict:
        checkpoint = {
            "n_steps": self._n_learning_steps,
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "online_q": self.online_q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "q_optim": self.q_optim.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optim": self.alpha_optim.state_dict(),
        }
        return checkpoint

    def restore(self, checkpoint: Dict):
        self._n_learning_steps = checkpoint["n_steps"]
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy_optim.load_state_dict(checkpoint["policy_optim"])
        self.online_q.load_state_dict(checkpoint["online_q"])
        self.q_optim.load_state_dict(checkpoint["q_optim"])
        self.target_q.load_state_dict(checkpoint["target_q"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha_optim.load_state_dict(checkpoint["alpha_optim"])

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
        self.policy = GaussianActionPolicy(self._model_config)
        self.policy.to(self._device)
        self.online_q = TwinQNetwork(self._model_config)
        self.online_q.to(self._device)
        self.target_q = TwinQNetwork(self._model_config)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.target_q.eval()
        self.target_q.to(self._device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self._policy_lr)
        self.q_optim = Adam(self.online_q.parameters(), lr=self._q_lr)
        self._target_entropy = -float(self._model_config["output_dim"])
        self.log_alpha = torch.zeros(1, device=self._device, requires_grad=True)
        self._alpha = self.log_alpha.detach().exp()
        self.alpha_optim = Adam([self.log_alpha], lr=self._entropy_lr)
