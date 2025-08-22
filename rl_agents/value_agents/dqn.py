from rl_agents.value_functions.dqn_function import DQNFunction
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.value_agents.value_agent import AbstractValueAgent

import torch

class DQNAgent(AbstractValueAgent):
    def __init__(
        self,
        nb_env: int,
        policy: AbstractPolicy,
        q_function : DQNFunction,
        train_every : int
    ):

        torch.nn.Module.__init__(self)
        AbstractValueAgent.__init__(self, q_function=q_function, nb_env=nb_env, policy=policy)
        self.q_function = q_function
        self.train_every = train_every

        assert isinstance(self.q_function, DQNFunction), "q_function must be from class DQNFunction, or inherit from it"


    def sample(self):
        return self.q_function.replay_memory.sample(agent=self, batch_size=self.q_function.batch_size, training=False)

    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval."
        
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        
        self.q_function.trainer.replay_memory.store(agent=self, **kwargs)

    def train_agent(self) -> float:
        super().train_agent()
        
        if self.step % self.train_every == 0:
            return self.q_function.train_service(agent = self)