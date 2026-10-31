import gymnasium as gym
from stable_baselines3 import PPO


class FixedRewardWrapper(gym.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self.prev_distance = None

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    # Reset distance tracking on new episode
    # obs[0] is x position, obs[1] is y position
    self.prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    # 1. Calculate current distance to the landing pad (0,0)
    current_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5

    # 2. Progress Reward: Reward the AI if it moved closer to the pad compared to last step
    distance_improvement = self.prev_distance - current_distance
    reward += distance_improvement * 10.0  # Scale the progress reward
    self.prev_distance = current_distance

    # 3. Fuel/Action Penalty: Small penalty if it fires engines, encouraging efficiency
    if action != 0:  # Action 0 is doing nothing
      reward -= 0.05

    return obs, reward, terminated, truncated, info


# Initialize and wrap the environment
base_env = gym.make("LunarLander-v3")
env = FixedRewardWrapper(base_env)

# Train a model with the proper reward structure
model = PPO("MlpPolicy", env, verbose=1)
print(
    "🛡️ Training agent with robust reward shaping (progress + efficiency"
    " penalties)..."
)
model.learn(total_timesteps=400_000)

model.save("ppo_fixed_lander")
print("✅ Fixed model saved as ppo_fixed_lander.zip")