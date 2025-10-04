from rl_agents.service import AgentService
from rl_agents.agent import Agent
from rl_agents.policies.policy import Policy, ContinuousPolicy, StochasticPolicy
from rl_agents.value_functions.value import Op
from rl_agents.utils.wrapper import Wrapper
from rl_agents.utils.check import is_instance, assert_is_instance

import torch
from gymnasium.spaces import Space

class GaussianNoiseWrapper(Wrapper, AgentService, Policy):
    def __init__(self, policy : Policy, std : float, **kwargs):
        super().__init__(**kwargs)
        self.policy : ContinuousPolicy = assert_is_instance(policy, ContinuousPolicy)
        self.std = std

        self.wrapped = self.policy # Respect Wrapper protocole
        self.is_stochastic = is_instance(self.policy, StochasticPolicy)

    def pick_action(self, state, op : Op, **kwargs):
        _return = self.policy.pick_action(state=state, **kwargs)  
        if not self.training or op != Op.ACTOR_PICK_ACTION:
            return _return
        
        if self.is_stochastic:  a, log_prob = _return
        else:                   a = _return
        a_noised = a + self.std * self.policy.scale * torch.randn_like(a)
        returned_a = a_noised.clamp(self.policy.low + self.policy.EPS, self.policy.high - self.policy.EPS) 

        if self.is_stochastic:  return returned_a, log_prob
        return returned_a

class OUNoiseWrapper(Wrapper, AgentService, Policy):
    """
    Ornstein–Uhlenbeck noise.
    """
    def __init__(self, policy: Policy, nb_env : int, action_space : Space, theta: float = 0.15, sigma: float = 0.2, relative: bool = True, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(policy, ContinuousPolicy), "policy must inherit from ContinuousPolicy"
        self.policy: ContinuousPolicy = policy
        self.nb_env = int(nb_env)
        self.theta = float(theta)
        self.relative = bool(relative)
        # buffers follow device/dtype automatically
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))
        self.register_buffer("x", torch.zeros(size=(nb_env, action_space.shape[0])))  # lazily resized to [B, A]
        
        self.wrapped: ContinuousPolicy = self.policy # Respect Wrapper proto
        self.is_stochastic = is_instance(self.policy, StochasticPolicy)

    def reset(self, agent : Agent, env_ids):
        """Reset OU state. If `mask` (shape [B]) is given, reset only those envs."""
        self.x[env_ids] = 0

    def pick_action(self, state: torch.Tensor, op : Op, **kwargs) -> torch.Tensor:
        _return = self.policy.pick_action(state=state, **kwargs)  

        if not self.training or op != Op.ACTOR_PICK_ACTION:
            return _return

        if self.is_stochastic:  a, log_prob = _return
        else:                   a = _return

        # per-dim sigma: relative -> scale by half-range; else absolute
        sigma = self.sigma * self.policy.scale if self.relative else self.sigma
        if sigma.ndim == 0:
            sigma = sigma.expand_as(a)

        # OU update (dt=1)
        self.x = self.x + self.theta * (0.0 - self.x) + sigma * torch.randn_like(a)
        a = a + self.x
        eps = getattr(self.policy, "EPS", 1E-6)
        
        returned_a = a.clamp(self.policy.low + eps, self.policy.high - eps)
        
        if self.is_stochastic: return returned_a, log_prob
        return returned_a