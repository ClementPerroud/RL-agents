from rl_agents.policies.policy import Policy, StochasticPolicy, DiscretePolicy, ContinuousPolicy
from collections import namedtuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent

import torch
import math
import gymnasium as gym
from abc import abstractmethod



class DiscreteStochasticPolicy(DiscretePolicy, StochasticPolicy):
    def __init__(self, core_net, action_space, **kwargs):
        super().__init__(core_net=core_net, action_space=action_space, **kwargs)

    def action_distributions(self, state : torch.Tensor) -> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities : torch.Tensor= self.forward(state)
        # action_probabilities : [batch, nb_action]
        assert action_probabilities.ndim == 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_probabilities.shape}"
        assert ((action_probabilities.sum(-1) - 1).abs() < 1E-6).all(), "The probabilities for each action must sum up to 1. Please apply softmax before returning the output"
        
        return (action_probabilities,)
    
    def evaluate_log_prob(self, action_distributions: torch.Tensor, action : torch.Tensor)-> torch.Tensor:
        action_probabilities, = action_distributions
        return (
            torch.take_along_dim(action_probabilities, action.long().unsqueeze(-1), dim = -1).squeeze(-1).clamp_min(1E-8)
        ).log()

    def pick_action(self, state : torch.Tensor, **kwargs)-> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities, = self.action_distributions(state=state)
        if self.training: action = torch.multinomial(input=action_probabilities, num_samples=1).squeeze(-1) # Pick the actions randomly following their given probabilities.
        else: action = action_probabilities.argmax(-1)
        
        log_prob = self.evaluate_log_prob(action_distributions= (action_probabilities,), action=action)
        return action.squeeze(-1), log_prob
        # shape [batch]

    def entropy_loss(self, action_probs : torch.Tensor):
        return torch.distributions.Categorical(probs = action_probs).entropy()


from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

_ContinuousActionDistribution = namedtuple("ContinuousActionDistribution", ["dist", "mean", "std"])

class ContinuousStochasticPolicy(ContinuousPolicy, StochasticPolicy):
    LOG_STD_MIN, LOG_STD_MAX = -2, 5
    EPS = 1E-6
    def __init__(self, action_space : gym.spaces.Box, core_net, *args, **kwargs):
        super().__init__(core_net = core_net, action_space=action_space, *args, **kwargs)
        self.low = torch.as_tensor(action_space.low, dtype = torch.float32)
        self.high = torch.as_tensor(action_space.high, dtype = torch.float32)
        self.scale = (self.high - self.low)/2
        self.loc = (self.high + self.low)/2
        self.head_mean = self.head
        self.head_log_std = torch.nn.LazyLinear(action_space.shape[0])
    
    def forward(self, states):
        x = self.core_net(states)
        return self.head_mean(x), self.head_log_std(x)

    def action_distributions(self, state) -> torch.Tensor:
        action_mean, action_log_std = self.forward(state) # type: ignore
        action_log_std = action_log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        if not torch.isfinite(action_mean).all(): raise RuntimeError("Actor mean has non-finite values")
        if not torch.isfinite(action_log_std).all(): raise RuntimeError("Actor log_std has non-finite values")

        assert action_mean.ndim >= 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_mean.shape}"
        assert action_mean.ndim == action_log_std.ndim, f"Both outputs (mean, log std) must have the same shape ({action_mean.shape} != {action_log_std.shape})"
        if action_mean.ndim == 1: # Make it -> [batch, 1] because nb_action = 1
            action_mean.unsqueeze_(-1)
            action_log_std.unsqueeze_(-1)
        return action_mean, action_log_std
    
    def evaluate_log_prob(self, action_distributions : tuple[torch.Tensor, torch.Tensor], action : torch.Tensor)-> torch.Tensor:
        mean, log_std = action_distributions
        std = log_std.exp()

        # map action back to (-1,1) then to u-space
        a_tanh = ((action - self.loc) / (self.scale + self.EPS)).clamp(-1 + self.EPS, 1 - self.EPS)
        u = self._atanh(a_tanh)

        base = torch.distributions.Normal(mean, std)
        log_prob_u = base.log_prob(u).sum(dim=-1)

        corr = (torch.log(1 - a_tanh.pow(2) + self.EPS)).sum(dim=-1)
        corr += torch.log(self.scale + self.EPS).sum()
        return log_prob_u - corr

    def pick_action(self, state : torch.Tensor, **kwargs):
        # state : [batch, state_shape ...]
        action_mean, action_log_std = self.action_distributions(state=state, **kwargs)

        # action_mean : [batch, nb_action], action_log_std [batch, nb_actions]
        action_std = action_log_std.exp()
        u = action_mean + action_std * torch.randn_like(action_mean)  # reparam/sample
        action = self._squash(u) # [1,A]
        
        log_prob = self.evaluate_log_prob(action_distributions= (action_mean, action_log_std), action=action)
        return action, log_prob
        # shape [batch]

    def entropy_loss(self, mean : torch.Tensor, log_std : torch.Tensor):
        return (0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std).sum(dim=-1)