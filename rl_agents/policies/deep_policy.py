from rl_agents.policies.policy import AbstractPolicy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

import torch
import math
import gymnasium as gym
from abc import abstractmethod

class AbstractDeepPolicy(AbstractPolicy):
    def __init__(self, policy_net : torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_net = policy_net

    @abstractmethod
    def action_distributions(self, agent: 'AbstractAgent', state : torch.Tensor):
        ...
    
    @abstractmethod
    def evaluate_log_prob(self, agent: 'AbstractAgent', action_distributions : torch.Tensor, action : torch.Tensor):
        ...
    
    @abstractmethod
    def entropy_loss(self, action_probs : torch.Tensor):
        ...
    


class DiscreteDeepPolicy(AbstractDeepPolicy):
    def action_distributions(self, state : torch.Tensor) -> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities : torch.Tensor= self.policy_net.forward(state)
        # action_probabilities : [batch, nb_action]
        assert action_probabilities.ndim == 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_probabilities.shape}"
        assert ((action_probabilities.sum(-1) - 1).abs() < 1E-6).all(), "The probabilities for each action must sum up to 1. Please apply softmax before returning the output"
        
        return (action_probabilities,)
    
    def evaluate_log_prob(self, action_distributions: torch.Tensor, action : torch.Tensor)-> torch.Tensor:
        action_probabilities, = action_distributions
        return (
            torch.take_along_dim(action_probabilities, action.long().unsqueeze(-1), dim = -1).squeeze(-1).clamp_min(1E-8)
        ).log()

    def pick_action(self, state : torch.Tensor)-> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities, = self.action_distributions(state=state)
        action = torch.multinomial(input=action_probabilities, num_samples=1).squeeze(-1) # Pick the actions randomly following their given probabilities.
        
        log_prob = self.evaluate_log_prob(action_distributions= (action_probabilities,), action=action)
        return action.squeeze(-1), log_prob
        # shape [batch]

    def entropy_loss(self, action_probs : torch.Tensor):
        return torch.distributions.Categorical(probs = action_probs).entropy().mean()

class ContinuousDeepPolicy(AbstractDeepPolicy):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  1.0

    def __init__(self, action_space: gym.spaces.Box, policy_net, *args, **kwargs):
        super().__init__(policy_net=policy_net, *args, **kwargs)

        low  = torch.as_tensor(action_space.low,  dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)

        # Some envs expose +/-inf; don't build an AffineTransform for those.
        self.register_buffer("_finite_bounds", torch.isfinite(low).all() & torch.isfinite(high).all())
        if self._finite_bounds:
            self.register_buffer("low",   low)
            self.register_buffer("high",  high)
            self.register_buffer("scale", (self.high - self.low) / 2)
            self.register_buffer("loc",   (self.high + self.low) / 2)
        self.epsilon = 1e-4

    def action_distributions(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy_net(state)  # type: ignore
        if mean.ndim == 1:
            mean = mean.unsqueeze(-1)
            log_std = log_std.unsqueeze(-1)
        assert mean.ndim >= 2 and mean.ndim == log_std.ndim, "bad policy_net output shapes"

        # clamp and sanitize
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise RuntimeError(f"Non-finite policy outputs: "
                               f"mean[{mean.min().item():.3g},{mean.max().item():.3g}], "
                               f"log_std[{log_std.min().item():.3g},{log_std.max().item():.3g}]")
        return mean, log_std

    def _dist(self, mean: torch.Tensor, log_std: torch.Tensor):
        base = torch.distributions.Normal(mean, log_std.exp())
        if getattr(self, "_finite_bounds").item():
            td = torch.distributions.TransformedDistribution(
                base,
                [torch.distributions.TanhTransform(cache_size=1),
                 torch.distributions.AffineTransform(loc=self.loc, scale=self.scale)]
            )
            return torch.distributions.Independent(td, 1)
        else:
            # unbounded actions: no squashing/affine
            return torch.distributions.Independent(base, 1)

    def evaluate_log_prob(self, action_distributions, action):
        action = action.clamp(self.low + self.epsilon, self.high - self.epsilon)
        mean, log_std = action_distributions
        dist = self._dist(mean, log_std)
        return dist.log_prob(action)

    def pick_action(self, state: torch.Tensor):
        mean, log_std = self.action_distributions(state=state)
        dist = self._dist(mean, log_std)
        action  = dist.rsample()
        # keep a hair away from hard bounds to avoid -inf jacobians at exactly ±1 after affine
        if getattr(self, "_finite_bounds").item():
            action = action.clamp(self.low + self.epsilon, self.high - self.epsilon)
            
        log_prob = dist.log_prob(action)
        if log_prob.isnan().sum() > 0:
            print("CATCH ", log_prob)
            
        return action, log_prob

    def entropy_loss(self, mean, log_std):
        ent = torch.distributions.Normal(mean, log_std.exp()).entropy()
        return ent.sum(-1).mean()
