import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cas4160.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(parameters, learning_rate)

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        # action = None
        ###########
        # obs를 torch tensor로 변환
        obs_t = ptu.from_numpy(np.expand_dims(obs, axis=0)) # shape: (1, ob_dim)

        # forward로 distribution 얻음
        dist = self.forward(obs_t)

        # action sampling
        action_t = dist.sample()                            # shape: (1, ac_dim) or (1,)

        # numpy로 변환 후 squeeze해서 단일 action 반환
        action = ptu.to_numpy(action_t).squeeze(0)
        ###########

        return action

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # HINT: use torch.distributions.Categorical to define the distribution.
            # dist = None
            ###########
            logits = self.logits_net(obs)                    # shape: (batch_size, ac_dim)
            dist = distributions.Categorical(logits=logits)
            ###########
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # HINT: use torch.distributions.Normal to define the distribution.
            # dist = None
            ###########
            mean = self.mean_net(obs)                        # shape: (batch_size, ac_dim)
            std = torch.exp(self.logstd)                     # shape: (ac_dim,)
            dist = distributions.Normal(mean, std)
            ###########

        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert obs.shape[0] == actions.shape[0] == advantages.shape[0]

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        # HINT: don't forget to do `self.optimizer.step()`!
        # loss = None
        ###########
        # 1. forward pass -> distribution 얻기
        dist = self.forward(obs)

        # 2. log probability 계산
        logp = dist.log_prob(actions)           # discrete일 때 (batch_size,), continuous일 때 (batch, ac_dim)

        if logp.ndim == 2:                      # continuous case
            logp = logp.sum(dim=-1)

        # 3. Policy Gradient loss (negative because we do gradient ascent)
        #    L = - E[ advantage * log π(a|s) ]
        loss = - (advantages * logp).mean()

        # 4. gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ###########

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert old_logp.ndim == 1
        assert advantages.shape == old_logp.shape

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        # TODO: Implement the ppo update.
        # HINT: calculate logp first, and then caculate ratio and clipped loss.
        # HINT: ratio is the exponential of the difference between logp and old_logp.
        # HINT: You can use torch.clamp to clip values.
        # loss = None
        ###########
        # 1. forward pass -> new distribution
        dist = self.forward(obs)

        # 2. new log probability
        logp = dist.log_prob(actions)           # shape: (batch_size,)
        if logp.ndim == 2:                      # continuous case
            logp = logp.sum(dim=-1)

        # 3. probability ratio
        ratio = torch.exp(logp - old_logp)

        # 4. clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)

        # 5. PPO loss (we minimize the negative surrogate)
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = -surrogate.mean()

        # 6. gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ###########

        return {"PPO Loss": ptu.to_numpy(loss)}
