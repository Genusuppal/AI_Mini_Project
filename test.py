import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import os
from NeuralNetwork import NeuralNetwork

CURRENT_MAX_STEPS = 5000

# Fitness function to evaluate the performance of an agent
def evaluate_agent(agent, env, max_steps=CURRENT_MAX_STEPS):
	total_reward = 0
	state, info = env.reset()  # Reset the environment to get initial state
	state = state.flatten() / 255.0  # Normalize the state
	
	for step in range(max_steps):
		action = np.argmax(agent.forward(state) + np.random.randn(env.action_space.n) * 0.1)  # Choose action using the neural network
		next_state, reward, terminated, truncated, info = env.step(action)  # Act and get the reward
		state = next_state.flatten() / 255.0  # Update state and normalize
		total_reward += reward
		if terminated or truncated:
			break
	return total_reward

def run():
	# Register the environment with gymnasium
	gym.register_envs(ale_py)  # Ensure ALE is registered
	
	# Create the environment with render_mode="rgb_array" for video recording
	env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
	
	input_size = env.observation_space.shape[0] * env.observation_space.shape[1] * 3  # Flattened RGB screen
	output_size = env.action_space.n  # Number of possible actions (discrete)
	
	agent = None
	if os.path.exists("best_agent_weights.npy"):
		print("Resuming from saved weights...")
		agent = NeuralNetwork(input_size, output_size)
		agent.load_weights("best_agent_weights.npy")
	else:
		print("No saved weights found, exiting...")
		return
	
	# Record video of the best agent's performance for the current generation
	video_folder = "saved-video-folder"
	
	env_with_video = gym.wrappers.RecordVideo(
		env,
		episode_trigger=lambda num: True,
		video_folder=video_folder,
		name_prefix=f"test_agent"
	)
	
	fitness = evaluate_agent(agent, env_with_video, CURRENT_MAX_STEPS)  # Evaluates the agent during recording
	print(f"Agent Fitness Score = {fitness}")

	env_with_video.close()

	# Close the environment properly after use
	env.close()

if __name__ == "__main__":
	run()
