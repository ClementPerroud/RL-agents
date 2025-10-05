from rl_agents.service import AgentService
from rl_agents.policies.policy import StochasticPolicy
from rl_agents.value_functions.value import Op
from rl_agents.memory.experience import ExperienceLike
from rl_agents.memory.sampler import Sampler
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.trainers.dqn import DQNTrainer
from rl_agents.trainers.mixin.off_policy import OffPolicyTrainerMixin
from rl_agents.trainers.mixin.value_critic import QCriticTrainerMixin
from rl_agents.utils.check import assert_is_instance

from rl_agents.utils.mode import eval_mode, train_mode
from rl_agents.utils.distribution.distribution import Distribution, distribution_aware, debug_mode, distribution_mode

import torch
from typing import Union
from gymnasium.spaces import Space

class SACTrainer(OffPolicyTrainerMixin, QCriticTrainerMixin):
    def __init__(self,
            # DQN Loss Paramaters
            q_loss_fn : torch.nn.modules.loss._Loss, 
            q_optimizer : torch.optim.Optimizer,
            policy_optimizer : torch.optim.Optimizer,
            sampler : Sampler,
            train_every : int,
            batch_size : int,
            init_alpha : float,
            alpha_lr : float,
            action_space : Space,
            gamma : float,
            q_policy : StochasticPolicy,
            *args, **kwargs):
        super().__init__(sampler=sampler, train_every=train_every, batch_size=batch_size)
        self.dqn_trainer = DQNTrainer(
            loss_fn=q_loss_fn,
            optimizer=q_optimizer,
            sampler=sampler,
            train_every=train_every,
            batch_size=batch_size,
            gamma = gamma,
            q_policy= q_policy
        )
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer

        self.log_alpha = torch.tensor(float(init_alpha)).log().requires_grad_(True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(action_space.shape[0])


    def set_up_and_check(self, agent : "ActorCriticAgent"):
        super().set_up_and_check(agent)
        self.dqn_trainer.set_up_and_check(agent)
        
        self.actor : Union[StochasticPolicy, torch.nn.Module] = assert_is_instance(agent.actor, StochasticPolicy)
        self.critic : AgentService = agent.critic

    @distribution_aware
    def train_step(self, experience : ExperienceLike):

        # Critic Loss
        self.actor.requires_grad_(False)
        with eval_mode(self.actor), train_mode(self.critic):
            critic_loss = self.dqn_trainer.train_step(
                experience=experience,
                alpha = self.log_alpha.exp().detach()
            )
        self.actor.requires_grad_(True)
        
        # Actor Loss
        self.q_function.requires_grad_(False)
        self.actor.train()
        self.q_function.eval()

        with train_mode(self.actor), eval_mode(self.q_function):
            # with torch.autograd.set_detect_anomaly(True):
            self.policy_optimizer.zero_grad(set_to_none=True)
            pred_action, log_prob = self.actor.pick_action(state = experience.state, op=Op.DDPG_LOSS_ACTOR_PICK_ACTION)
            

            actor_q = self.q_function.Q(state=experience.state, action =pred_action, op=Op.DDPG_LOSS_CRITIC_Q)
            if isinstance(actor_q, Distribution):
                actor_q = actor_q.expectation()

            actor_loss = self.log_alpha.exp().detach()*log_prob.mean() - actor_q.mean()
            actor_loss.backward()

            self.policy_optimizer.step()
            self.q_function.requires_grad_(True)
        
        
        # Temperature Loss
        self.alpha_optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, log_prob = self.actor.pick_action(state=experience.state, op=Op.DDPG_LOSS_ACTOR_PICK_ACTION)

        alpha = self.log_alpha.exp()
        alpha_loss = -(alpha * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()

        self._alpha_losses.append(alpha_loss.item())
        if alpha_loss.item() > 1:
            print("Catch")

        return critic_loss.detach(), actor_loss.detach(), alpha_loss.detach()
    
    from collections import deque
    _alpha_losses = deque(maxlen=1000)

    def sample_pre_hook(self):
        return self.dqn_trainer.sample_pre_hook()