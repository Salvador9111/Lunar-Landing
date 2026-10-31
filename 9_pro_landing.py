import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment with rendering turned on
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Load your newly trained pro model
model = PPO.load("ppo_pro_lander")

print("🏆 Evaluating your Pro-Tier RL Agent...")

# 3. Test across 5 episodes
for episode in range(5):
  observation, info = env.reset()
  done = False
  score = 0

  while not done:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    done = terminated or truncated

  print(f"Episode {episode + 1} | Pro Score: {score:.2f}")

env.close()