from rl_agents.value_functions.dqn_function import DQN
from rl_agents.value_functions.value import Q, Trainable
from rl_agents.policies.policy import Policy
from rl_agents.value_agents.value_agent import AbstractValueAgent
from rl_agents.replay_memory.replay_memory import BaseReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import AbstractSampler
from rl_agents.utils.mode import eval_mode
import torch


class DQNAgent(AbstractValueAgent):
    def __init__(
        self,
        nb_env: int,
        policy: Policy,
        q_function : DQN,
        train_every : int,
        replay_memory : BaseReplayMemory,
        sampler : AbstractSampler,
        batch_size : int,
        optimizer : torch.optim.Optimizer,
        **kwargs
    ):

        assert isinstance(q_function, Q) and isinstance(q_function, Trainable), "q_function must implement Q and Trainable."

        if isinstance(replay_memory, MultiStepReplayMemory): q_function.gamma **= replay_memory.multi_step
        super().__init__(q_function=q_function, nb_env=nb_env, policy=policy, **kwargs)
        self.q_function = q_function
        self.train_every = train_every
        self.replay_memory = replay_memory
        self.sampler = sampler
        self.batch_size = batch_size
        self.optimizer = optimizer

    def store(self, **experience_kwargs):
        assert self.training, "Cannot store any memory during eval."
        
        for key, value in experience_kwargs.items():
            experience_kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: experience_kwargs[key] = experience_kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        
        indices = self.replay_memory.store(**experience_kwargs) # indices : [nb_env,]
        if indices is not None: self.sampler.store(experience = self.replay_memory[indices],**experience_kwargs)

    def train_agent(self) -> float:
        super().train_agent()
        
        if self.step % self.train_every == 0 and self.step > self.batch_size:
            # Training Q function
            loss = self.train_step()
            return loss

    def train_step(self):
        indices = self.sampler.sample(self.batch_size)
        experience = self.replay_memory[indices]

        self.optimizer.zero_grad()

        loss_input = self.q_function.compute_loss_input(experience=experience)
        with eval_mode(self), torch.no_grad():
            loss_target = self.q_function.compute_loss_target(experience=experience)
        loss = self.q_function.loss_fn(loss_input, loss_target)

        loss = self._apply_weights(loss, self.sampler.compute_weights_from_indices(experience.indices))

        loss = loss.mean()
        loss.backward()

        self.optimizer.step()

        with torch.no_grad():
            self.sampler.update_experiences(
                agent = self, indices = experience.indices, td_errors = self.q_function.compute_td_errors(loss_input=loss_input, loss_target=loss_target)
            )
        return loss.item()
        
    def _apply_weights(self, loss : torch.Tensor, weight : torch.Tensor | float |None):
        # Handle weights
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            try: return loss * weight
            except BaseException as e: 
                raise ValueError(f"Could not apply weight from a loss of shape {loss.shape} and {weight.shape}. Initial Exception : {e}")
        return loss