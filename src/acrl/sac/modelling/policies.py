from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal

MAX_STDEV = 2
MIN_STDEV = -20


def xavier_initialisation(layer: nn.Module, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ActionPolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        input_dim = config["input_dim"]
        feature_dim = config["feature_dim"]
        output_dim = config["output_dim"]
        self._model = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim),
        ).apply(xavier_initialisation)

    def __call__(self, x: Tensor) -> Tensor:
        return self._model(x)


class TwinActionPolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        config = config.copy()
        config["input_dim"] = config["input_dim"] + config["output_dim"]
        self._model_1 = ActionPolicy(config)
        self._model_2 = ActionPolicy(config)

    def __call__(self, actions: Tensor, states: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([states, actions], dim=1)
        return self._model_1(x), self._model_2(x)


class GaussianActionPolicy(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        config = config.copy()
        config["output_dim"] = 2 * config["output_dim"]
        self._model = ActionPolicy(config)

    def __call__(self, states: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        means, log_stdvs = torch.chunk(self._model(states), 2, dim=-1)
        log_stdvs = torch.clamp(log_stdvs, min=MIN_STDEV, max=MAX_STDEV)
        stdvs = log_stdvs.exp_()
        normals = Normal(means, stdvs)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)
        return actions, entropies, torch.tanh(means)
