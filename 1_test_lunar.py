import gymnasium as gym

# 1. Create the environment with render_mode="human" to pop up the game window
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Reset the environment to get the initial state
observation, info = env.reset()
print(
    "🚀 Initial 8-element observation vector:", observation
)

# 3. Run a simulation loop for 500 steps
for step_num in range(500):
  # Pick a completely random action from the 4 available choices
  action = env.action_space.sample()

  # Step the environment forward using that action
  observation, reward, terminated, truncated, info = env.step(action)

  # Print out the current observations and rewards to watch the data change
  print(
      f"Step {step_num} | Action: {action} | Y-Pos: {observation[1]:.2f} |"
      f" Reward: {reward:.2f}"
  )

  # If the rocket crashes or finishes an episode, reset it
  if terminated or truncated:
    print("💥 Rocket crashed or episode ended! Resetting environment...")
    observation, info = env.reset()

# Close the window cleanly when done
env.close()