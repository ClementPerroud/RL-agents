from rl_agents.service import AgentService
from rl_agents.policies.policy import Policy
from rl_agents.value_functions.value import Op
from rl_agents.memory.memory import ExperienceSample
from rl_agents.memory.sampler import Sampler
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.trainers.dqn import DQNTrainer
from rl_agents.trainers.mixin.off_policy import OffPolicyTrainerMixin
from rl_agents.trainers.mixin.value_critic import QCriticTrainerMixin

from rl_agents.utils.mode import eval_mode, train_mode
from rl_agents.utils.distribution.distribution import Distribution, distribution_aware, debug_mode, distribution_mode

import torch
from typing import Union

class SACTrainer(OffPolicyTrainerMixin, QCriticTrainerMixin):
    def __init__(self,
            # DQN Loss Paramaters
            q_loss_fn : torch.nn.modules.loss._Loss, 
            q_optimizer : torch.optim.Optimizer,
            policy_optimizer : torch.optim.Optimizer,
            sampler : Sampler,
            train_every : int,
            batch_size : int,
            *args, **kwargs):
        super().__init__(sampler=sampler, train_every=train_every, batch_size=batch_size)
        self.dqn_trainer = DQNTrainer(loss_fn=q_loss_fn, optimizer=q_optimizer, sampler=sampler, train_every=train_every, batch_size=batch_size)
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer


    def set_up_and_check(self, agent : "ActorCriticAgent"):
        super().set_up_and_check(agent)
        self.dqn_trainer.set_up_and_check(agent)
        
        self.actor : Union[Policy, torch.nn.Module] = agent.actor
        self.critic : AgentService = agent.critic

    @distribution_aware
    def train_step(self, experience : ExperienceSample):

        # Critic Loss
        self.actor.requires_grad_(False)
        with eval_mode(self.actor), train_mode(self.critic):
            critic_loss = self.dqn_trainer.train_step(experience=experience)
        self.actor.requires_grad_(True)

        # Actor Loss
        self.q_function.requires_grad_(False)
        self.actor.train()
        self.q_function.eval()

        with train_mode(self.actor), eval_mode(self.q_function):
            # with torch.autograd.set_detect_anomaly(True):
            self.policy_optimizer.zero_grad(set_to_none=True)
            pred_action = self.actor.pick_action(state = experience.state, op=Op.DDPG_LOSS_ACTOR_PICK_ACTION)
            

            actor_q = self.q_function.Q(state=experience.state, action =pred_action, op=Op.DDPG_LOSS_CRITIC_Q)
            if isinstance(actor_q, Distribution):
                actor_q = actor_q.expectation()

        actor_loss = -actor_q.mean()


        actor_loss.backward()

        self.policy_optimizer.step()
        self.q_function.requires_grad_(True)

        return critic_loss, actor_loss.item()