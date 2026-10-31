import gymnasium as gym
from stable_baselines3 import PPO


# 1. Create a Custom Wrapper to modify the rewards
class RewardHackingWrapper(gym.Wrapper):

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    # Get the standard outputs from the physics engine
    observation, reward, terminated, truncated, info = self.env.step(action)

    # --- REWARD HACKING INJECTION ---
    # Extract the tilt angle (index 4) and vertical velocity (index 3)
    angle = abs(observation[4])
    vertical_velocity = abs(observation[3])

    # If the rocket is roughly upright and moving slowly, give it a free reward boost!
    if angle < 0.1 and vertical_velocity < 0.5:
      reward += 2.0  # Unintended incentive: "Just hover here forever and you'll get rich!"
    # ---------------------------------

    return observation, reward, terminated, truncated, info


# 2. Initialize environment and wrap it
base_env = gym.make("LunarLander-v3")
env = RewardHackingWrapper(base_env)

# 3. Train a new quick model with this broken reward system
model = PPO("MlpPolicy", env, verbose=1)
print(
    "🧪 Training agent with a hacked/flawed reward function (shorter training"
    " for quick demo)..."
)
model.learn(total_timesteps=300_000)

# Save the hacked model
model.save("ppo_hacked_lander")
print("✅ Hacked model saved as ppo_hacked_lander.zip")