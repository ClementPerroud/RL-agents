import numpy as np
from gymnasium.spaces import Box

from rl_agents.replay_memory.replay_memory import MultiStepReplayMemory


class DummyAgent:
    pass


def test_multistep_replay_memory_store_handles_agent():
    obs_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    memory = MultiStepReplayMemory(
        length=10, observation_space=obs_space, num_envs=1, multi_step=2, gamma=0.9
    )
    agent = DummyAgent()

    state0 = np.array([[0.0]], dtype=np.float32)
    state1 = np.array([[0.1]], dtype=np.float32)
    state2 = np.array([[0.2]], dtype=np.float32)
    action = np.array([0], dtype=np.int64)
    reward = np.array([1.0], dtype=np.float32)
    not_done = np.array([False])
    done = np.array([True])

    memory.store(
        agent=agent,
        state=state0,
        action=action,
        next_state=state1,
        reward=reward,
        done=not_done,
    )
    assert memory.size() == 0

    memory.store(
        agent=agent,
        state=state1,
        action=action,
        next_state=state2,
        reward=reward,
        done=not_done,
    )
    assert memory.size() == 1

    memory.store(
        agent=agent,
        state=state2,
        action=action,
        next_state=state2,
        reward=reward,
        done=done,
    )
    # flush remaining steps on episode end
    assert memory.size() == 3