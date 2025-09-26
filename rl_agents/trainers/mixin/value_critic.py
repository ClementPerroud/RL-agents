from rl_agents.service import AgentService
from rl_agents.trainers.agent import BaseAgentTrainer
from rl_agents.trainers.mixin.off_policy import OffPolicyTrainerMixin
from rl_agents.memory.sampler import Sampler
from rl_agents.value_functions.value import V, Q, Trainable
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.utils.hidden_modules import HiddenModulesUtilsMixin
from rl_agents.utils.assert_check import assert_is_instance

from typing import Union

class VCriticTrainerMixin(
        HiddenModulesUtilsMixin,
        BaseAgentTrainer
    ):

    @property
    def value_function(self) -> Union[V, AgentService]:
        """Hidden reference to avoid infinite recursion with torch recursive method (like .to(), .train() ...)"""
        return self.hidden.value_function

    def set_up_and_check(self, agent):
        super().set_up_and_check(agent)
        # CRITIC CHECK : Check if the critic is a trainable Q function.
        assert_is_instance(self.agent, ActorCriticAgent)
        assert_is_instance(self.agent.critic, V)
        self.hidden.value_function = self.agent.critic

class QCriticTrainerMixin(
        HiddenModulesUtilsMixin,
        BaseAgentTrainer):
    
    @property
    def q_function(self) -> Union[Q, V, AgentService]:
        """Hidden reference to avoid infinite recursion with torch recursive method (like .to(), .train() ...)"""
        return self.hidden.q_function

    def set_up_and_check(self, agent):
        super().set_up_and_check(agent)
        # CRITIC CHECK : Check if the critic is a trainable Q function.
        assert_is_instance(self.agent, ActorCriticAgent)
        assert_is_instance(self.agent.critic, V)
        assert_is_instance(self.agent.critic, Q)
        self.hidden.q_function = self.agent.critic
        
        assert_is_instance(self.agent.trainer, OffPolicyTrainerMixin)
        self.sampler : Sampler = self.agent.trainer.sampler
