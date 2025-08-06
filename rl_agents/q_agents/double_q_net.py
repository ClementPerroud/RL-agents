from rl_agents.agent import AbstractAgent
from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from copy import deepcopy
import torch
from typing import Callable


class DoubleQNNProxy(AbstractDeepQNeuralNetwork):
    def __init__(
        self,
        q_net_generator : 'AbstractDeepQNeuralNetwork',
        tau: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.q_net: AbstractDeepQNeuralNetwork = q_net_generator()
        self.q_net_target: AbstractDeepQNeuralNetwork = q_net_generator().requires_grad_(False)

        self.q_net = self.q_net.connect(self)
        self.q_net_target = self.q_net_target.connect(self)
        

    def forward(self, *args, target: bool = False, **kwargs):
        if target:
            return self.q_net_target.forward(*args, target=True, **kwargs)
        return self.q_net.forward(*args, target=False, **kwargs)

    @torch.no_grad()
    def update(self, agent: AbstractAgent):
        if agent.step % self.tau == 0:
            self._copy_weights()

    def _copy_weights(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())
