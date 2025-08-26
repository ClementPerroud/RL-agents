from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.replay_memory.sampler import RandomSampler
from rl_agents.value_functions.v_function import AbstractVFunction
from rl_agents.policies.deep_policy import AbstractDeepPolicy
from rl_agents.replay_memory.rollout_memory import RolloutMemory
from rl_agents.replay_memory.policy_training_memory import PPOTrainingMemory
from rl_agents.policy_agents.advantage_function import BaseAdvantageFunction
from rl_agents.utils.collates import do_nothing_collate

import torch
import numpy as np
from copy import deepcopy
import gymnasium as gym
import itertools    

class PPOLoss(AgentService):
    def __init__(self, epsilon: float, entropy_loss_coeff: float, value_clip_eps: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_loss_coeff = entropy_loss_coeff
        self.epsilon = epsilon
        self.value_clip_eps = value_clip_eps

    def forward(self, agent: AbstractAgent, policy: AbstractDeepPolicy, experience):
        mean_logit = policy.action_distributions(agent=agent, state=experience.state)
        logp_new = policy.evaluate_log_prob(agent=agent, action_distributions=mean_logit, action=experience.action)

        logp_old = experience.log_prob.detach()
        if logp_old.isnan().sum() > 0:
            print("CATCH")

        ratio = (logp_new - logp_old).clamp(min=-5, max = 2).exp()
        adv = experience.advantage.detach()

        unclipped = ratio * adv
        clipped = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * adv
        policy_obj = torch.minimum(unclipped, clipped).mean()

        entropy = policy.entropy_loss(*mean_logit)          # averaged

        # We only return -(policy + entropy); value loss is computed outside with returns
        return -(policy_obj + self.entropy_loss_coeff * entropy)



class A2CAgent(AbstractPolicyAgent):
    """Advantage Actor-Critic Agent"""
    def __init__(self,
            nb_env : int,
            policy : AbstractDeepPolicy,
            # policy : AbstractDeepPolicy, -> Actor -> Policy ?
            # value_function : AbstractVFunction, -> Critic -> AdvantageFunction
            advantage_function : BaseAdvantageFunction,
            policy_loss : PPOLoss,

            rollout_period : int,
            epoch_per_rollout : int,
            batch_size : int,
            values_loss_coeff : float,

            observation_space : gym.spaces.Space,
            action_space : gym.spaces.Space,

        ):
        super().__init__(nb_env = nb_env, policy = policy)
        assert isinstance(self.policy, AbstractDeepPolicy), "policy must be a DeepPolicy."
        self.policy : AbstractDeepPolicy

        self.advantage_function = advantage_function

        self.rollout_period = rollout_period
        self.epoch_per_rollout = epoch_per_rollout
        self.rollout_memory = RolloutMemory(max_length=rollout_period, observation_space=observation_space, action_space=action_space)
        self.policy_loss = policy_loss

        self.batch_size = batch_size

        self._opt_parameters = list({id(p): p for p in itertools.chain(
            self.policy.parameters(),
            self.advantage_function.parameters())
        }.values())
        self.optimizer = torch.optim.Adam(
            params=self._opt_parameters,
            lr = 3E-4,
            eps=1E-5
        )
        self.values_loss_coeff = values_loss_coeff

    @torch.no_grad()
    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval. Please set your agent to TRAINING mode."
        
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        # Adding log_prob        
        if kwargs["log_prob"].isnan().sum() > 0:
            print("CATCH ", kwargs["log_prob"])
        self.rollout_memory.store(agent=self, **kwargs)


    def train_agent(self):
        super().train_agent()

        mean_loss = None
        if self.step % self.rollout_period == 0:
            # 1 - Precomputations
            self.advantage_function.compute(agent = self)
            
            # 2 - Start the training
            # Normalize advantage
            adv = self.rollout_memory["advantage"][:len(self.rollout_memory)]
            self.rollout_memory["advantage"] = (adv - adv.mean()) / (adv.std().clamp_min(1e-8))

            data_loader = torch.utils.data.DataLoader(
                dataset=self.rollout_memory,
                batch_size=self.batch_size,
                shuffle=True, drop_last=False,
                collate_fn=do_nothing_collate,
            )
            losses = []
            for _ in range(self.epoch_per_rollout):
                for experience in data_loader:
                    self.optimizer.zero_grad()

                    # Actor loss (PPO clipped)
                    policy_loss = self.policy_loss(agent=self, policy=self.policy, experience=experience)

                    # Critic loss to GAE returns
                    loss_inputs = self.advantage_function.value_function.compute_loss_inputs(experience=experience)
                    value_loss = self.advantage_function.value_function.loss_fn(*loss_inputs).mean()

                    loss = policy_loss + self.values_loss_coeff * value_loss
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()

                    # torch.nn.utils.clip_grad_norm_(self.policy.policy_net.parameters(), max_norm=0.5)
                    # torch.nn.utils.clip_grad_norm_(self.advantage_function.value_function.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    losses.append(loss.item())
            
            # 3 - After training
            mean_loss = np.mean(losses)
            self.rollout_memory.reset()            
        return mean_loss
