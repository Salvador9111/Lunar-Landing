import gymnasium as gym
from stable_baselines3 import PPO


class FixedRewardWrapper(gym.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self.prev_distance = None

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    self.prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    current_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    distance_improvement = self.prev_distance - current_distance
    reward += distance_improvement * 10.0
    self.prev_distance = current_distance

    # Minor action penalty to encourage fuel efficiency
    if action != 0:
      reward -= 0.03

    return obs, reward, terminated, truncated, info


# 1. Initialize environment and wrap it
base_env = gym.make("LunarLander-v3")
env = FixedRewardWrapper(base_env)

# 2. Configure professional hyperparameters and a larger neural network
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)  # Larger brain: 2 hidden layers with 256 neurons each

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0002,  # Slightly dialed-in learning rate
    n_steps=2048,
    batch_size=64,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

print(
    "🚀 Starting pro-tier training (1M steps + larger network architecture)..."
)
model.learn(total_timesteps=1_000_000)

# Save the master model
model.save("ppo_pro_lander")
print("✅ Pro model saved as ppo_pro_lander.zip")