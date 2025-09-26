from rl_agents.service import AgentService
from typing import Protocol, runtime_checkable

@runtime_checkable
class Wrapper(Protocol):
    wrapped : AgentService
    def __init__(self, **kwargs): super().__init__(**kwargs)