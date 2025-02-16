import pickle
from pathlib import Path
from typing import Dict

import torch
from acrl.sac.sac import SoftActorCritic
from acrl.buffer.replay_buffer import ReplayBuffer


class Checkpointer:
    def __init__(
        self,
        checkpoint_path: Path,
        model: SoftActorCritic,
        replay_buffer: ReplayBuffer,
    ):
        self._path = checkpoint_path
        self._model = model
        self._replay_buffer = replay_buffer

    def checkpoint(self, n_steps: int):
        self._reset_checkpoint(n_steps)
        self._checkpoint_policy()
        self._checkpoint_q()
        self._checkpoint_alpha()
        self._save_model_checkpoint()
        self._save_replay_buffer()

    def _reset_checkpoint(self, n_steps: int):
        self._checkpoint = {"n_steps": n_steps}

    def _checkpoint_policy(self):
        policy_state = {
            "policy": self._model.policy.state_dict(),
            "policy_optim": self._model.policy_optim.state_dict(),
        }
        self._checkpoint.update(policy_state)

    def _checkpoint_q(self):
        q_state = {
            "online_q": self._model.online_q.state_dict(),
            "target_q": self._model.target_q.state_dict(),
            "q_optim": self._model.q_optim.state_dict(),
        }
        self._checkpoint.update(q_state)

    def _checkpoint_alpha(self):
        alpha_state = {
            "log_alpha": self._model.log_alpha.state_dict(),
            "alpha_optim": self._model.alpha_optim.state_dict(),
        }
        self._checkpoint.update(alpha_state)

    def _save_model_checkpoint(self):
        n_steps = self._checkpoint["n_steps"]
        path = self._path.joinpath(f"{n_steps}.tar")
        torch.save(self._checkpoint, path)

    def _save_replay_buffer(self):
        n_steps = self._checkpoint["n_steps"]
        path = self._path.joinpath(f"{n_steps}-replay_buffer.pkl")
        with path.open("wb") as file:
            pickle.dump(self._replay_buffer, file)

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(f"{path}.tar")
        self._load_policy(checkpoint)
        self._load_q(checkpoint)
        self._load_alpha(checkpoint)
        self._load_replay_buffer(path)

    def _load_policy(self, checkpoint: Dict):
        self._model.policy.load_state_dict(checkpoint["policy"])
        self._model.policy_optim.load_state_dict(checkpoint["policy_optim"])

    def _load_q(self, checkpoint: Dict):
        self._model.online_q.load_state_dict(checkpoint["q_optim"])
        self._model.q_optim.load_state_dict(checkpoint["online_q"])
        self._model.target_q.load_state_dict(checkpoint["target_q"])

    def _load_alpha(self, checkpoint: Dict):
        self._model.log_alpha.load_state_dict(checkpoint["log_alpha"])
        self._model.alpha_optim.load_state_dict(checkpoint["alpha_optim"])

    def _load_replay_buffer(self, path: Path) -> ReplayBuffer:
        path = Path(f"{path}-replay_buffer.pkl")
        with path.open("rb") as file:
            self._replay_buffer = pickle.load(file)
