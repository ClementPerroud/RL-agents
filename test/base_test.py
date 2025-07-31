from rl_agents.action_model.model import OTPActionModel
from rl_agents.replay_memory import ReplayMemory
from rl_agents.dqn import BaseDQNgent

import torch

nb_env = 5
action_model = OTPActionModel(nb_env=nb_env, action = 0)
replay_memory = ReplayMemory(length = 1E5, nb_features= 6)
agent = BaseDQNgent(
    gamma=0.99,
    tau = 1000,
    replay_memory= None,
    nb_env= nb_env,
    action_model= action_model,
)


class TestModule1(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dense = torch.nn.Linear(32, 32)

class TestModule2(torch.nn.Module):
    def __init__(self, test_module1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_module1 = test_module1
        self.dense = torch.nn.Linear(32, 32)
    
test_module = TestModule2(test_module1= TestModule1())

print([_ for _ in agent.children()])
print([_ for _ in test_module.children()])