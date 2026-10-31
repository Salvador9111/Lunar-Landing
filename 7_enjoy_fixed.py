import gymnasium as gym
from stable_baselines3 import PPO

# 1. Load the base environment with rendering turned on
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Load your robustly trained fixed model
model = PPO.load("ppo_fixed_lander")

print("🎯 Playing back your optimized, non-hacked agent...")

# 3. Test it across a few episodes
for episode in range(3):
  observation, info = env.reset()
  done = False
  score = 0

  while not done:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    done = terminated or truncated

  print(f"Episode {episode + 1} | Score: {score:.2f}")

env.close()