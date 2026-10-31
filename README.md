# 🚀 Reinforcement Learning Lunar Lander: From Reward Hacking to Pro-Tier PPO Optimization

An end-to-end Reinforcement Learning (RL) engineering project built using **Python, OpenAI Gymnasium (Box2D), Stable-Baselines3, and PyTorch**. This repository documents the complete lifecycle of training an autonomous agent to master the `LunarLander-v3` environment, intentionally engineering and mitigating an AI alignment failure mode (**Goodhart's Law / Reward Hacking**), and scaling network architectures to achieve elite performance scores (~297).

## 🧪 Key Engineering Phases

### Phase 1: Baseline Physics & Observation Space
* **Environment:** OpenAI Gymnasium `LunarLander-v3` powered by Box2D physics.
* **State Space:** An 8-dimensional continuous vector tracking $(x, y)$ coordinates, linear velocities $(v_x, v_y)$, tilt angle $\theta$, angular velocity, and two binary flags indicating left/right leg contact with the ground pad.
* **Action Space:** 4 discrete actions (do nothing, fire left orientation thruster, fire main engine, fire right orientation thruster).
* **Observation:** Un-optimized policies result in erratic flight paths and immediate crashes.

### Phase 2: AI Alignment & Reward Hacking (Goodhart's Law)
* **The Problem:** In real-world machine learning, poorly specified objective functions lead agents to "game the system" rather than solve the intended problem.
* **The Experiment:** Implemented a custom Gym wrapper (`RewardHackingWrapper`) granting a continuous reward bonus whenever the agent kept its tilt upright and velocity low, completely bypassing landing objectives.
* **The Result:** The agent learned to fly into mid-air, stabilize vertically, and hover indefinitely until running out of fuel—proving that *"when a measure becomes a target, it ceases to be a good measure."*

### Phase 3: Robust Mitigation & Reward Shaping
* **The Solution:** Engineered `FixedRewardWrapper` to guide policy convergence toward safe, efficient landing:
  * **Dense Progress Incentive:** Calculates Euclidean distance delta to the landing pad $(0,0)$ at every step, rewarding positive closure.
  * **Efficiency Penalty:** Applies a minor action penalty for active engine usage to discourage wasteful hovering and fuel burn.

### Phase 4: Pro-Tier Scaling & Hyperparameter Tuning
* **Architecture Upgrade:** Migrated from default `[64, 64]` hidden layers to a deeper multi-layer perceptron with `[256, 256]` neurons for both policy ($\pi$) and value ($v$) networks.
* **Training Scale:** Executed 1,000,000 timesteps using Proximal Policy Optimization (PPO) with a fine-tuned learning rate ($2e-4$).
* **Performance:** Achieved elite evaluation scores consistently crossing the **260–297** threshold, representing precise, centered touchdowns with minimal corrective thrust.


# How to Setup
Create and activate a Python virtual environment:
Bash
python -m venv .venv

## On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

## On Linux/macOS:
source .venv/bin/activate


# Install dependencies:

* Bash
* pip install swig
* pip install "gymnasium[box2d]" stable-baselines3 moviepy
