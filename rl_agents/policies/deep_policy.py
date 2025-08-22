from rl_agents.policies.policy import AbstractPolicy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

import torch
import math
import gymnasium as gym
from abc import abstractmethod

class AbstractDeepPolicy(
    AbstractPolicy
    ):
    @abstractmethod
    def action_distributions(self, agent: 'AbstractAgent', state : torch.Tensor):
        ...
    
    @abstractmethod
    def evaluate_action_log_likelihood(self, agent: 'AbstractAgent', action_distributions : torch.Tensor, action : torch.Tensor):
        ...
    
    @abstractmethod
    def entropy_loss(self, action_probs : torch.Tensor):
        ...
    
    def __init__(self, policy_net : torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_net = policy_net
        self.last_log_prob = 0


class DiscreteDeepPolicy(AbstractDeepPolicy):
    def action_distributions(self, agent: 'AbstractAgent', state : torch.Tensor) -> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities : torch.Tensor= self.policy_net.forward(state)
        # action_probabilities : [batch, nb_action]
        assert action_probabilities.ndim == 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_probabilities.shape}"
        assert ((action_probabilities.sum(-1) - 1).abs() < 1E-6).all(), "The probabilities for each action must sum up to 1. Please apply softmax before returning the output"
        
        return (action_probabilities,)
    
    def evaluate_action_log_likelihood(self, agent: 'AbstractAgent', action_distributions: torch.Tensor, action : torch.Tensor)-> torch.Tensor:
        action_probabilities, = action_distributions
        return (
            torch.take_along_dim(action_probabilities, action.long().unsqueeze(-1), dim = -1).squeeze(-1).clamp_min(1E-8)
        ).log()

    def pick_action(self, agent, state : torch.Tensor)-> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities, = self.action_distributions(agent=agent, state=state)
        action = torch.multinomial(input=action_probabilities, num_samples=1) # Pick the actions randomly following their given probabilities.
        
        self.last_log_prob = self.evaluate_action_log_likelihood(agent=agent, action_distributions= (action_probabilities,), action=action)
        return action.squeeze(-1)
        # shape [batch]

    def entropy_loss(self, action_probs : torch.Tensor):
        return torch.distributions.Categorical(probs = action_probs).entropy().mean()

class ContinuousDeepPolicy(AbstractDeepPolicy):
    def __init__(self, action_space : gym.spaces.Box, policy_net, *args, **kwargs):
        self.low = torch.as_tensor(action_space.low, dtype = torch.float32)
        self.high = torch.as_tensor(action_space.high, dtype = torch.float32)
        self.scale = (self.high - self.low)/2
        self.loc = (self.high + self.low)/2

        super().__init__(policy_net=policy_net, *args, **kwargs)

    def action_distributions(self, agent: 'AbstractAgent', state) -> torch.Tensor:
        action_mean, action_log_std = self.policy_net.forward(state) # type: ignore
        action_mean : torch.Tensor
        action_log_std : torch.Tensor

        assert action_mean.ndim >= 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_mean.shape}"
        assert action_mean.ndim == action_log_std.ndim, f"Both outputs (mean, log std) must have the same shape ({action_mean.shape} != {action_log_std.shape})"
        if action_mean.ndim == 1: # Make it -> [batch, 1] because nb_action = 1
            action_mean.unsqueeze_(-1)
            action_log_std.unsqueeze_(-1)
        return action_mean, action_log_std
    
    def evaluate_action_log_likelihood(self, agent: 'AbstractAgent', action_distributions : tuple[torch.Tensor, torch.Tensor], action : torch.Tensor)-> torch.Tensor:
        mean, log_std = action_distributions
        std = log_std.exp()

        # map action back to (-1,1) then to u-space
        a_tanh = ((action - self.loc) / self.scale).clamp(-0.999999, 0.999999)
        u = self._atanh(a_tanh)

        base = torch.distributions.Normal(mean, std)
        log_prob_u = base.log_prob(u).sum(dim=-1)

        corr = (torch.log(1 - a_tanh.pow(2) + 1e-6)).sum(dim=-1)
        corr += torch.log(self.scale).sum()
        return log_prob_u - corr

    @staticmethod
    def _atanh(x, eps=1e-6):
        x = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
    def _squash(self, u):
        a = torch.tanh(u)
        return self.loc + a * self.scale, a 

    def pick_action(self, agent: 'AbstractAgent', state : torch.Tensor):
        # state : [batch, state_shape ...]
        action_mean, action_log_std = self.action_distributions(agent=agent, state=state)

        # action_mean : [batch, nb_action], action_log_std [batch, nb_actions]
        action_std = action_log_std.exp()
        u = action_mean + action_std * torch.randn_like(action_mean)  # reparam/sample
        action, _ = self._squash(u)            # [1,A]
        
        self.last_log_prob = self.evaluate_action_log_likelihood(agent=agent, action_distributions= (action_mean, action_log_std), action=action)
        return action
        # shape [batch]

    def entropy_loss(self, mean : torch.Tensor, log_std : torch.Tensor):
        return torch.distributions.Normal(mean, log_std.exp()).entropy().mean()