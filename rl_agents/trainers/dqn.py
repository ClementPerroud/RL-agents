from rl_agents.memory.sampler import Sampler, WeightableSampler, UpdatableSampler
from rl_agents.memory.memory import ExperienceSample
from rl_agents.memory.replay_memory import MultiStepReplayMemory
from rl_agents.trainers.mixin.off_policy import OffPolicyTrainerMixin
from rl_agents.trainers.mixin.value_critic import QCriticTrainerMixin
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.utils.distribution.distribution import distribution_aware
import torch


class DQNTrainer(OffPolicyTrainerMixin, QCriticTrainerMixin):
    def __init__(self,
            # DQN Loss Paramaters
            loss_fn : torch.nn.modules.loss._Loss, 
            optimizer : torch.optim.Optimizer,
            sampler : Sampler,
            train_every : int,
            batch_size : int,
            *args, **kwargs):
        super().__init__(sampler=sampler, train_every=train_every, batch_size=batch_size)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "none"


    def set_up_and_check(self, agent : "ActorCriticAgent"):
        super().set_up_and_check(agent)
        if isinstance(self.replay_memory, MultiStepReplayMemory): self.q_function.gamma **= self.replay_memory.multi_step

    @distribution_aware
    def train_step(self, experience : ExperienceSample):

        self.optimizer.zero_grad(set_to_none=True)

        q_loss = self.compute_loss(
            experience=experience
        )

        q_loss.backward()
        self.optimizer.step()

        return q_loss.item()

    def compute_loss(self, experience : ExperienceSample) -> torch.Tensor:
        loss_input = self.q_function.compute_loss_input(experience=experience)
        with torch.no_grad():
            loss_target = self.q_function.compute_loss_target(experience=experience)

        loss : torch.Tensor = self.loss_fn(loss_input, loss_target)
        if isinstance(self.sampler, WeightableSampler): loss = self.sampler.apply_weights(loss=loss, indices=experience.indices)
        loss = loss.mean()

        if isinstance(self.sampler, UpdatableSampler):
            with torch.no_grad():
                self.sampler.update_experiences(
                    agent = self, indices = experience.indices, td_errors = self.q_function.compute_td_errors(loss_input=loss_input, loss_target=loss_target)
                )
        
        return loss