# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- Activate Python virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
- Install dependencies: `pip install -r requirements.txt`

### Testing
- Run all tests: `pytest test/`
- Run specific test: `pytest test/sumtree_test.py`
- Run with verbose output: `pytest -v test/`

### Running Examples
- Debug/example scripts are located in the `debug/` folder
- Run PPO discrete example: `python debug/ppo_1_env_lunarlander_discrete.py`
- Run DQN example: `python debug/dqn_1_env_cartpole.py`
- Run Categorical DQN example: `python debug/categorical_dqn_1_env_lunarlander_1.py`

## Architecture Overview

RainbowTorch is a modular reinforcement learning library built on PyTorch. The architecture follows a service-oriented design pattern where components are connected via the `AgentService` system.

### Core Components

**Agent Hierarchy (`rl_agents/agent.py`)**:
- `AbstractAgent`: Base class for all RL agents with service discovery and state management
- Agents inherit from both `AbstractPolicy` and `AgentService` for modular composition
- Two main agent types: Policy-based (`policy_agents/`) and Value-based (`q_agents/`)

**Service System (`rl_agents/service.py`)**:
- `AgentService`: Base class enabling modular composition via `.connect()` method
- Services automatically discover and update sub-services in a tree structure
- Components like policies, value functions, and memory systems are all services

**Agent Types**:
- **PPO Agent** (`policy_agents/ppo_agent.py`): Proximal Policy Optimization with GAE advantage estimation
- **DQN Agent** (`q_agents/dqn.py`): Deep Q-Network with experience replay and target networks
- **Categorical DQN**: Distributional RL implementation

### Key Architectural Patterns

**Policy System** (`policies/`):
- `AbstractPolicy`: Base interface for all action selection policies
- `DeepPolicy`: Neural network-based policies for continuous/discrete actions  
- `EpsilonGreedyProxy`: Wrapper for exploration strategies

**Memory Systems** (`replay_memory/`):
- `AbstractReplayMemory`: Base class for experience storage
- `RolloutMemory`: For on-policy algorithms (PPO)
- `MultiStepReplayMemory`: For off-policy with n-step returns
- `PrioritizedSampler`: Priority-based experience replay

**Value Functions** (`value_functions/`):
- Modular Q-functions and V-functions
- Support for distributional RL (Categorical DQN)
- Advantage function estimation (GAE)

**Training Infrastructure** (`trainers/`):
- `Trainer`: Handles optimization, loss computation, and weight updates
- `Trainable`: Interface for components that can be trained
- Automatic loss weighting and gradient handling

### Service Connection Pattern

Components connect using: `component.connect(parent_service)`, which:
1. Registers the component as a sub-service
2. Enables automatic updates during agent steps
3. Allows service discovery throughout the hierarchy

Example connection flow:
```python
agent = PPOAgent(policy=policy, value_function=value_function, ...)
# Internally: value_function.connect(agent), policy.connect(agent)
```

### Environment Integration

- Built for Gymnasium environments
- Supports both discrete and continuous action spaces  
- Handles single and multi-environment setups
- Automatic tensor conversion and batching

### Testing Structure

Tests are organized by component in `test/`:
- Data structure tests (e.g., `sumtree_test.py`)
- Agent-specific functionality tests
- Memory system validation tests