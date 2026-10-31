import gymnasium as gym
from stable_baselines3 import PPO

# Load the hacked environment and model
env = gym.make(
    "LunarLander-v3", render_mode="human"
)  # Note: test on base env to see real behavior
model = PPO.load("ppo_hacked_lander")

print("👀 Watching the reward-hacked agent...")
observation, info = env.reset()

for _ in range(1000):
  action, _ = model.predict(observation, deterministic=True)
  observation, reward, terminated, truncated, info = env.step(action)

  if terminated or truncated:
    observation, info = env.reset()

env.close()