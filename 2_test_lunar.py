import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment for training (no need for render_mode="human" here, training headless is much faster!)
env = gym.make("LunarLander-v3")

# 2. Initialize the PPO (Proximal Policy Optimization) agent
# "MlpPolicy" tells it to use a standard Multi-Layer Perceptron (neural network)
model = PPO("MlpPolicy", env, verbose=1)

print("🚀 Starting AI training... This will take a few minutes.")

# 3. Train the model for 1 million timesteps
# (1 million steps usually takes about 15-20 minutes on a modern laptop CPU)
model.learn(total_timesteps=1_000_000)

# 4. Save the trained brain to your disk
model.save("ppo_lunar_lander")
print("✅ Training complete! Model saved successfully as ppo_lunar_lander.zip.")