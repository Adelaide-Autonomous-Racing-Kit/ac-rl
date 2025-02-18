import os
from pathlib import Path
from glob import glob
from typing import Dict, List

import torch
import numpy as np
from loguru import logger
from acrl.sac.sac import SoftActorCritic
from acrl.buffer.replay_buffer import ReplayBuffer


class Checkpointer:
    def __init__(
        self,
        config: Dict,
        checkpoint_path: Path,
        model: SoftActorCritic,
        replay_buffer: ReplayBuffer,
    ):
        self._path = checkpoint_path
        self._model = model
        self._replay_buffer = replay_buffer
        self._is_resuming = config["resume"]
        self._n_checkpoints = config["n_to_keep"]
        self._setup_folder()

    def checkpoint(self, n_actions: int):
        self._save_model_checkpoint(n_actions)
        self._save_replay_buffer(n_actions)
        self._maybe_delete_old_checkpoints()
        self._log_checkpoint(n_actions)

    def _save_model_checkpoint(self, n_actions: int):
        checkpoint = self._model.serialise()
        checkpoint["n_actions"] = n_actions
        path = self._path.joinpath(f"{n_actions}.tar")
        torch.save(checkpoint, path)

    def _save_replay_buffer(self, n_actions: int):
        path = self._path.joinpath(f"{n_actions}-replay_buffer")
        replay_buffer_state = self._replay_buffer.serialise_buffer()
        np.save(path, replay_buffer_state, allow_pickle=True)

    def _maybe_delete_old_checkpoints(self):
        model_files = glob(os.path.join(self._path, "*.tar"))
        buffer_files = glob(os.path.join(self._path, "*.npy"))
        if len(model_files) > self._n_checkpoints:
            to_delete = sort_files_by_age(model_files)[0]
            os.remove(to_delete)
            to_delete = sort_files_by_age(buffer_files)[0]
            os.remove(to_delete)

    def _log_checkpoint(self, n_actions: int):
        path = self._path.joinpath(f"{n_actions}")
        logger.info(f"Checkpoint saved to {path}")

    def load(self, checkpoint_name: str) -> int:
        path = f"{self._path}/{checkpoint_name}"
        n_actions = self._restore_model(path)
        self._restore_replay_buffer(path)
        self._log_restoration(checkpoint_name)
        return n_actions

    def _restore_model(self, path: str):
        checkpoint = torch.load(f"{path}.tar", weights_only=False)
        self._model.restore(checkpoint)
        return checkpoint["n_actions"]

    def _restore_replay_buffer(self, path: str):
        path = f"{path}-replay_buffer.npy"
        replay_buffer_state = np.load(path, allow_pickle=True).item()
        self._replay_buffer.restore(replay_buffer_state)

    def _log_restoration(self, checkpoint_name: str):
        path = self._path.joinpath(f"{checkpoint_name}")
        logger.info(f"Training restored from checkpoint {path}")

    def _setup_folder(self):
        if self._is_resuming:
            return
        try:
            self._path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            message = f"The checkpoint folder {self._path} already exists. "
            message += "If you want to overwrite the folder please delete it manually."
            message += "If you want to resume from a checkpoint set resume to True."
            logger.error(message)


def sort_files_by_age(files: List):
    return sorted(files, key=lambda x: os.stat(x).st_mtime)
