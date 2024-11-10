import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import os
from NeuralNetwork import NeuralNetwork

# Hyperparameters for the Genetic Algorithm
POPULATION_SIZE = 50  # Increase for better diversity
CURR_GENERATION = 40
NUM_GENERATIONS = 100  # Increase for better evolution
INITIAL_MAX_STEPS = 200  # Starting steps for each generation
END_MAX_STEPS = 2000  # Final step count for a generation
STEPS_INCREMENT = 40  # Steps increment for each generation
CURRENT_MAX_STEPS = INITIAL_MAX_STEPS

# Fitness function to evaluate the performance of an agent
def evaluate_agent(agent, env, max_steps=CURRENT_MAX_STEPS):
	total_reward = 0
	state, info = env.reset()  # Reset the environment to get initial state
	state = state.flatten() / 255.0  # Normalize the state
	
	for step in range(max_steps):
		action = np.argmax(agent.forward(state) + np.random.randn(env.action_space.n) * 0.5)  # Choose action using the neural network
		next_state, reward, terminated, truncated, info = env.step(action)  # Act and get the reward
		state = next_state.flatten() / 255.0  # Update state and normalize
		total_reward += reward
		if terminated or truncated:
			break
	return total_reward

# Create initial population of agents
def create_population(pop_size, input_size, output_size):
	return [NeuralNetwork(input_size, output_size) for _ in range(pop_size)]

# Tournament Selection: select parents for crossover
def tournament_selection(population, fitness_scores, tournament_size=4):
	selected = []
	for _ in range(2):  # We need two parents
		tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
		tournament.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness
		selected.append(tournament[0][0])  # Select the best individual
	return selected

# Crossover: Two-point crossover
def crossover(parent1, parent2):
	# Crossover by selecting random points and exchanging parts of the weights
	child1 = parent1.copy()
	child2 = parent2.copy()
	
	crossover_point = random.randint(1, parent1.weights.shape[0] - 1)
	child1.weights[crossover_point:] = parent2.weights[crossover_point:]
	child2.weights[crossover_point:] = parent1.weights[crossover_point:]
	
	return child1, child2

# Evolve the population by selecting parents, applying crossover and mutation
def evolve_population(population, env, fitness_scores):
	# Select the top 20% as elites
	elite_size = POPULATION_SIZE // 5
	elites = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:elite_size]
	elites = [elite[0] for elite in elites]
	
	new_population = elites  # Start the new population with elites

	# Perform crossover and mutation to fill the rest of the population
	while len(new_population) < POPULATION_SIZE:
		parent1, parent2 = tournament_selection(population, fitness_scores)
		child1, child2 = crossover(parent1, parent2)
		child1.mutate()
		child2.mutate()
		new_population.extend([child1, child2])
	
	# Ensure population size is maintained by truncating any excess
	new_population = new_population[:POPULATION_SIZE]
	
	return new_population

def run():
    # Register the environment with gymnasium
    gym.register_envs(ale_py)  # Ensure ALE is registered
    
    # Create the environment with render_mode="rgb_array" for video recording
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    
    input_size = env.observation_space.shape[0] * env.observation_space.shape[1] * 3  # Flattened RGB screen
    output_size = env.action_space.n  # Number of possible actions (discrete)
    
    # Check if we are resuming from a saved agent's weights
    best_agent_of_generation = None
    if os.path.exists("best_agent_weights.npy"):
        print("Resuming from saved weights...")
        best_agent_of_generation = NeuralNetwork(input_size, output_size)
        best_agent_of_generation.load_weights("best_agent_weights.npy")
        population = create_population(POPULATION_SIZE - 1, input_size, output_size)
        population.append(best_agent_of_generation)
    else:
        # Create the initial population of agents
        population = create_population(POPULATION_SIZE, input_size, output_size)
    
    fitness_history = []  # Track fitness over generations
    
    for generation in range(CURR_GENERATION, NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")
        
        # Increase MAX_STEPS dynamically for each generation
        CURRENT_MAX_STEPS = min(INITIAL_MAX_STEPS + (generation * STEPS_INCREMENT), END_MAX_STEPS)
        print(f"Max Steps for Generation {generation + 1}: {CURRENT_MAX_STEPS}")
        
        # Evaluate each agent's performance in the current generation
        fitness_scores = []
        best_agent_of_generation = None
        best_fitness_of_generation = -float('inf')
        
        for idx, agent in enumerate(population):
            print(f"Evaluating Agent {idx + 1} in Generation {generation + 1}")
            fitness_score = evaluate_agent(agent, env, CURRENT_MAX_STEPS)
            fitness_scores.append(fitness_score)
            
            if fitness_score > best_fitness_of_generation:
                best_fitness_of_generation = fitness_score
                best_agent_of_generation = agent
        
        # After evaluating all agents, evolve the population
        population = evolve_population(population, env, fitness_scores)
        
        # Plot the best agent's performance for this generation
        print(f"Best agent of Generation {generation + 1}: Fitness Score = {best_fitness_of_generation}")
        fitness_history.append(best_fitness_of_generation)
        
        # Record video of the best agent's performance for the current generation
        video_folder = "saved-video-folder"
        
        env_with_video = gym.wrappers.RecordVideo(
            env,
            episode_trigger=lambda num: True,
            video_folder=video_folder,
            name_prefix=f"generation_{generation}_best_agent"
        )
        
		# Evaluates the agent during recording
        evaluate_agent(best_agent_of_generation, env_with_video, CURRENT_MAX_STEPS)
        env_with_video.close()
        
        # Save the weights of the best agent after each generation
        best_agent_of_generation.save_weights("best_agent_weights.npy")
        
    # Plot the fitness history over generations
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Reward)')
    plt.title('Fitness History of Genetic Algorithm')
    plt.show()

    # Close the environment properly after use
    env.close()

if __name__ == "__main__":
	run()
