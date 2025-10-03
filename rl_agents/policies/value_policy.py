from rl_agents.policies.policy import Policy
from rl_agents.value_functions.value import Q, Op
from rl_agents.utils.hidden_modules import HiddenModulesUtilsMixin
from rl_agents.utils.assert_check import assert_is_instance
import numpy as np
import torch



# Use for testing purpose
class DiscreteBestQValuePolicy(HiddenModulesUtilsMixin, Policy):
    def __init__(self, q_fn : Q, **kwargs):
        super().__init__(**kwargs)
        assert_is_instance(q_fn, Q)
        self.hidden.q_fn = q_fn

    def pick_action(self, state: torch.Tensor, op = Op.ACTOR_PICK_ACTION, **kwargs):
        return torch.argmax(
            self.hidden.q_fn.Q_per_action(state = state, op = op, **kwargs), 
            dim=-1
        ).long()