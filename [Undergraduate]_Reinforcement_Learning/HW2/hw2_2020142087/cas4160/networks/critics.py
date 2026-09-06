import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cas4160.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        # return None
        ###########
        return self.network(obs)
        ###########

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        assert obs.ndim == 2
        assert q_values.ndim == 1

        # TODO: update the critic using the observations and q_values
        # loss = None
        ###########
        # 1. critic의 예측값 얻기
        values = self.forward(obs)                    # shape: (batch_size, 1)

        # 2. target q_values의 shape -> (batch_size, 1)
        targets = q_values.unsqueeze(-1)              # shape: (batch_size, 1)

        # 3. Mean Squared Error loss
        # (value function은 Q-value를 regress하도록 학습)
        loss = F.mse_loss(values, targets)

        # 4. optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ###########

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }
