import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment with render_mode="human" so you can watch it fly
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Load your trained model from disk
model = PPO.load("ppo_lunar_lander")

print("🎮 Playing back trained agent...")

# 3. Run a few test episodes to watch the trained AI land
for episode in range(5):
  observation, info = env.reset()
  done = False
  score = 0

  while not done:
    # Ask the trained model what action to take based on the current observation
    action, _states = model.predict(observation, deterministic=True)

    # Take the step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward

    done = terminated or truncated

  print(f"Episode {episode + 1} finished with total score: {score:.2f}")

env.close()