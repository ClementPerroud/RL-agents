from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent

class PPOAgent(AbstractPolicyAgent):
    def __init__(self, nb_env, policy, device = None):
        super().__init__(nb_env, policy, device)


if __name__ == "__main__":
    agent = PPOAgent(1, None)
