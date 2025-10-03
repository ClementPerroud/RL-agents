from rl_agents.memory.sampler import Sampler, WeightableSampler, UpdatableSampler
from rl_agents.memory.experience import ExperienceLike
from rl_agents.memory.replay_memory import MultiStepReplayMemory
from rl_agents.trainers.mixin.off_policy import OffPolicyTrainerMixin
from rl_agents.trainers.mixin.value_critic import QCriticTrainerMixin
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.utils.distribution.distribution import Distribution, distribution_aware, distribution_mode
from rl_agents.value_functions.value import Q, V, Trainable, Op
from rl_agents.policies.policy import Policy
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.utils.assert_check import assert_is_instance

import torch


class DQNTrainer(
            OffPolicyTrainerMixin,
            QCriticTrainerMixin
        ):
    def __init__(self,
            # DQN Loss Paramaters
            loss_fn : torch.nn.modules.loss._Loss, 
            optimizer : torch.optim.Optimizer,
            sampler : Sampler,
            train_every : int,
            batch_size : int,
            gamma : float,
            q_policy : Policy,
            *args, **kwargs):
        super().__init__(sampler=sampler, train_every=train_every, batch_size=batch_size)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "none"
        self.gamma = gamma
        self.hidden.q_policy = q_policy

        self._last_sampler_update_step = 0

    @property
    def q_policy(self) -> DiscreteBestQValuePolicy: return self.hidden.q_policy

    def set_up_and_check(self, agent : "ActorCriticAgent"):
        super().set_up_and_check(agent)
        assert_is_instance(self.hidden.q_policy, Policy)

        self.gamma_computation = self.gamma
        if isinstance(self.replay_memory, MultiStepReplayMemory): self.gamma_computation **= self.replay_memory.multi_step


    def sample_pre_hook(self):
        # Compute weights for new experiences (if needed)
        if isinstance(self.sampler, UpdatableSampler) and len(self.replay_memory) > 0:
            n = len(self.replay_memory)
            # Retrieve new experience
            last_indices = torch.arange(self._last_sampler_update_step, n) 
            last_experiences = self.replay_memory[last_indices]
            # Compute their weights
            _, td_errors = self.compute_loss(experience=last_experiences, return_loss=False, return_td_errors=True)
            self.sampler.update_experiences(indices=last_experiences.indices, weights = td_errors)
            self._last_sampler_update_step = n

    def train_step(self, experience : ExperienceLike):
        # 1 - Train DQN
        self.optimizer.zero_grad(set_to_none=True)
        q_loss, td_errors = self.compute_loss(experience=experience, return_td_errors=isinstance(self.sampler, UpdatableSampler))
        q_loss.backward()
        self.optimizer.step()

        # 2 - Update Sampler weights (if needed)
        if isinstance(self.sampler, UpdatableSampler):
            self.sampler.update_experiences(indices = experience.indices, weights = td_errors)

        return q_loss.item()

    @distribution_aware
    def compute_loss(self, experience : ExperienceLike, return_loss = True, return_td_errors = True) -> torch.Tensor:
        loss_input = self.q_function.Q(experience.state, experience.action, op = Op.DQN_LOSS_INPUT_Q)

        with torch.no_grad():
            mask_end = 1 - experience.done.float()
            with distribution_mode(False):
                best_action = self.q_policy.pick_action(state = experience.next_state, op = Op.DQN_LOSS_TARGET_PICK_ACTION)
            q_values =self.q_function.Q(state=experience.next_state, action=best_action, op=Op.DQN_LOSS_TARGET_Q)

            loss_target = experience.reward + mask_end * self.gamma_computation * q_values
        
        loss = None
        if return_loss:
            loss : torch.Tensor = self.loss_fn(loss_input, loss_target)
            
            if loss.isnan().any(): raise ValueError("Nan Loss")
            
            if isinstance(self.sampler, WeightableSampler): loss = self.sampler.apply_weights(loss=loss, indices=experience.indices)
            loss = loss.mean()
            
            if loss.isnan().any(): raise ValueError("Nan Loss")
            self._debug_last_loss = loss
            self._debug_last_experience = experience

        td_errors = None
        if return_td_errors:
            with torch.no_grad():
                loss_input_td, loss_target_td = loss_input, loss_target
                if isinstance(loss_input_td, Distribution): loss_input_td = loss_input_td.expectation()
                if isinstance(loss_target_td, Distribution): loss_target_td = loss_target_td.expectation()
                td_errors = (loss_input_td - loss_target_td).abs()
        
        return loss, td_errors
