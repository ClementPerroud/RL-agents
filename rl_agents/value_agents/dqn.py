from rl_agents.value_functions.dqn_function import DQNFunction
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.value_agents.value_agent import AbstractValueAgent
from rl_agents.replay_memory.replay_memory import BaseReplayMemory
from rl_agents.replay_memory.sampler import AbstractSampler
import torch


class DQNAgent(AbstractValueAgent):
    def __init__(
        self,
        nb_env: int,
        policy: AbstractPolicy,
        q_function : DQNFunction,
        train_every : int,
        replay_memory : BaseReplayMemory,
        sampler : AbstractSampler,
        batch_size : int,
        optimizer : torch.optim.Optimizer
    ):

        torch.nn.Module.__init__(self)
        AbstractValueAgent.__init__(self, q_function=q_function, nb_env=nb_env, policy=policy)
        self.q_function = q_function
        self.train_every = train_every
        self.replay_memory = replay_memory
        self.sampler = sampler
        self.batch_size = batch_size
        self.optimizer = optimizer

        def collate_fn(data): return data
        self.dataloader = torch.utils.data.DataLoader(dataset=self.replay_memory, sampler=self.sampler, collate_fn=collate_fn, batch_size=batch_size)
        self._dataloader_iter = iter(self.dataloader)
        assert isinstance(self.q_function, DQNFunction), "q_function must be from class DQNFunction, or inherit from it"


    def sample(self, n_samples : int):
        return self.replay_memory.sample(agent=self, batch_size=n_samples, training=False)

    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval."
        
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        
        self.replay_memory.store(agent=self, **kwargs)
        self.sampler.store(agent=self, **kwargs)


    def train_step_from_dataloader(self):
        try:
            experience = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self.dataloader)
            experience = next(self._dataloader_iter)
        
        self.optimizer.zero_grad()

        q_loss = self.q_function.get_loss(agent=self, experience=experience)
        q_loss = self._apply_weights(q_loss, self.sampler.compute_weights_from_indices(experience.indices))

        q_loss = q_loss.mean()
        q_loss.backward()

        self.optimizer.step()

        return q_loss.item()

            
    def train_agent(self) -> float:
        super().train_agent()
        
        if self.step % self.train_every == 0 and self.step > self.batch_size:
            # Training Q function
            loss = self.train_step_from_dataloader()
            return loss

    def _apply_weights(self, loss : torch.Tensor, weight : torch.Tensor | float |None):
        # Handle weights
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            try: return loss * weight
            except BaseException as e: 
                raise ValueError(f"Could not apply weight from a loss of shape {loss.shape} and {weight.shape}. Initial Exception : {e}")
        return loss