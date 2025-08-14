from rl_agents.policies.policy import AbstractPolicy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

import torch
import math
from abc import abstractmethod

class AbstractDeepPolicy(
    torch.nn.Module,
    AbstractPolicy
    ):
    @abstractmethod
    def action_distributions(self, agent: 'AbstractAgent', state : torch.Tensor, training : bool):
        ...
    
    @abstractmethod
    def evaluate_action_log_likelihood(self, agent: 'AbstractAgent', action_distributions : torch.Tensor, action : torch.Tensor, training : bool):
        ...
    
    
    def __init__(self, policy_net : torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_net = policy_net


class DiscreteDeepPolicy(AbstractDeepPolicy):
    def action_distributions(self, agent: 'AbstractAgent', state : torch.Tensor, training : bool) -> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities : torch.Tensor= self.policy_net.forward(state, training = training)
        # action_probabilities : [batch, nb_action]
        assert action_probabilities.ndim == 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_probabilities.shape}"
        assert (action_probabilities.sum(-1) - 1).abs() < 1E-6, "The probabilities for each action must sum up to 1. Please apply softmax before returning the output"
        return action_probabilities
    
    def evaluate_action_log_likelihood(self, agent: 'AbstractAgent', action_distributions: torch.Tensor, action : torch.Tensor, training : bool)-> torch.Tensor:
        return (action_distributions[action.long()] + 1E-8).log()

    def pick_action(self, agent, state : torch.Tensor, training : bool)-> torch.Tensor:
        # state : [batch, state_shape ...]
        action_probabilities : torch.Tensor= self.action_distributions(agent=agent, state=state, training=training)
        actions = torch.multinomial(input=action_probabilities, num_samples=1) # Pick the actions randomly following their given probabilities.
        return actions 
        # shape [batch]


class ContinuousDeepPolicy(AbstractDeepPolicy):

    def action_distributions(self, agent: 'AbstractAgent', state, training) -> torch.Tensor:
        action_mean, action_log_std = self.policy_net.forward(state, training = training) # type: ignore
        action_mean : torch.Tensor
        action_log_std : torch.Tensor

        assert action_mean.ndim > 2, f"The policy_net ouput must be of shape [batch, nb_action] while current shape is {action_mean.shape}"
        assert action_mean.ndim == action_log_std.ndim, f"Both outputs (mean, log std) must have the same shape ({action_mean.shape} != {action_log_std.shape})"
        if action_mean.ndim == 1: # Make it -> [batch, 1] because nb_action = 1
            action_mean.unsqueeze_(-1)
            action_log_std.unsqueeze_(-1)
        return action_mean, action_log_std
    
    c = - math.log(2*math.pi) / 2
    def evaluate_action_log_likelihood(self, agent: 'AbstractAgent', action_distributions : tuple[torch.Tensor, torch.Tensor], action : torch.Tensor, training : bool)-> torch.Tensor:
        mean_prob, log_std_prob = action_distributions 
        log_likelihood = self.c - log_std_prob -  (action - mean_prob).pow(2) / (2 * (log_std_prob.exp().pow(2)))
        return log_likelihood

    def pick_action(self, agent: 'AbstractAgent', state : torch.Tensor, training : bool):
        # state : [batch, state_shape ...]
        action_mean, action_log_std = self.action_distributions(agent=agent, state=state, training=training)
        
        # action_mean : [batch, nb_action], action_log_std [batch, nb_actions]
        actions = torch.normal(mean= action_mean, std = action_log_std.exp()) # Pick the actions randomly following their given probabilities.
        return actions
        # shape [batch]