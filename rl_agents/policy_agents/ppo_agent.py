from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.policies.deep_policy import AbstractDeepPolicy
from rl_agents.replay_memory.replay_memory import ReplayMemory
import torch
from copy import deepcopy

class PPOAgent(
        torch.nn.Module,
        AbstractPolicyAgent):
    
    def __init__(self,
            nb_env,
            policy,
            device = None,
            rollout_period = 2048,
        ):
        super().__init__(nb_env, policy, device)
        assert isinstance(self.policy, AbstractDeepPolicy), "policy must be a DeepPolicy."
        self.policy : AbstractDeepPolicy
        self.old_policy : AbstractDeepPolicy = deepcopy(policy)

        self.rollout_period = rollout_period
        self.rollout_memory = ReplayMemory(length=rollout_period * 2)
    

    def train_agent(self):
        if self.step % self.T == 0:

            state, action = ()
            ratio = self.policy.evaluate_action_log_likelihood(
                    self,
                    action_distributions=self.policy.action_distributions(self, state),
                    action = action
                )

if __name__ == "__main__":
    agent = PPOAgent(1, None)
