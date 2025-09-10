import numpy as np
from gymnasium.spaces import Box

from rl_agents.memory.replay_memory import MultiStepReplayMemory
from rl_agents.memory.sampler import RandomSampler


class DummyAgent:
    def __init__(self):
        self.step = 0
        self.training = True


def test_multistep_replay_memory_store_handles_agent():
    obs_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    memory = MultiStepReplayMemory(
        max_length=10, observation_space=obs_space, nb_env=1, multi_step=2, gamma=0.9, sampler=RandomSampler()
    )
    agent = DummyAgent()

    state0 = np.array([0.0], dtype=np.float32)
    state1 = np.array([0.1], dtype=np.float32)
    state2 = np.array([0.2], dtype=np.float32)
    action = np.array(0, dtype=np.int64)
    reward = np.array(1.0, dtype=np.float32)
    not_done = np.array(False)
    done = np.array(True)
    not_truncated = np.array(False)


    memory.store(
        agent=agent,
        state=np.expand_dims(state0, 0),
        action=np.expand_dims(action, 0),
        next_state=np.expand_dims(state1, 0),
        reward=np.expand_dims(reward, 0),
        done=np.expand_dims(not_done, 0),
        truncated=np.expand_dims(not_truncated, 0)
    )
    assert len(memory) == 0

    memory.store(
        agent=agent,
        state=np.expand_dims(state1, 0),
        action=np.expand_dims(action, 0),
        next_state=np.expand_dims(state2, 0),
        reward=np.expand_dims(reward, 0),
        done=np.expand_dims(not_done, 0),
        truncated=np.expand_dims(not_truncated, 0)
    )
    assert len(memory) == 1

    memory.store(
        agent=agent,
        state=np.expand_dims(state2, 0),
        action=np.expand_dims(action, 0),
        next_state=np.expand_dims(state2, 0),
        reward=np.expand_dims(reward, 0),
        done=np.expand_dims(done, 0),
        truncated=np.expand_dims(not_truncated, 0)
    )
    # flush remaining steps on episode end
    assert len(memory) == 3