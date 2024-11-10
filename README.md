# Genetic Algorithm for Training a Pong Agent

This project implements a Genetic Algorithm (GA) to train an agent to play **Pong** using a simple **Neural Network**. The agent evolves over generations, with the best-performing agents being selected for crossover and mutation to produce better offspring. The goal is to maximize the agent's total reward, which is tracked and plotted over generations.

### Key Features:
- **Genetic Algorithm**: The population evolves using selection, crossover, and mutation.
- **Neural Network**: A simple feedforward neural network is used to predict actions based on the environment's state.
- **Pong Game**: The agent is trained to play the Pong game from the Atari 2600 emulator using the **Gymnasium** library.
- **Video Recording**: The best agent's gameplay for each generation is recorded and saved.
- **Checkpointing**: The best agent's weights are saved after each generation, allowing for resumption of training from the last saved weights.

## Requirements

To run the project, you'll need Python and the following libraries:

- `gymnasium` (to access Learning Environment)
- `ale-py` (with Atari environments)
- `numpy`
- `matplotlib`
- `random`
- `os`

You can install these dependencies using `pip`:

```bash
pip install gymnasium ale-py numpy matplotlib
```

## Project Structure

```
.
├── saved-video-folder/             # Folder where gameplay videos are saved
├── best_agent_weights.npy         # File where the best agent's weights are saved
├── genetic_algorithm_pong.py      # Main Python script for running the GA
├── README.md                      # This readme file
└── requirements.txt               # List of required Python packages
```

## Usage

### Running the Genetic Algorithm:

To start the training, simply run the `genetic_algorithm_pong.py` script:

```bash
python genetic_algorithm_pong.py
```

This will:
- Train a population of agents over multiple generations.
- Save the best agent's weights after each generation.
- Record gameplay videos of the best agent's performance.
- Plot the fitness history (total reward over generations) after the training completes.

### Resuming from Saved Weights:

If you want to resume training from a previously saved agent, make sure the `best_agent_weights.npy` file is present in the directory. The training will continue from the last saved agent’s weights.

To resume training, simply run the script again:

```bash
python genetic_algorithm_pong.py
```

The script will automatically detect the saved weights and load them to continue training.

### Customization:

You can modify several hyperparameters for the genetic algorithm to control the evolution process. The following hyperparameters are defined at the top of the `genetic_algorithm_pong.py` file:

- `POPULATION_SIZE`: Number of agents in each generation (default: 50)
- `MUTATION_RATE`: Probability of mutation (default: 0.1)
- `MUTATION_SCALE`: Scale of mutation noise (default: 0.05)
- `NUM_GENERATIONS`: Number of generations to run the GA (default: 50)
- `INITIAL_MAX_STEPS`: Maximum number of steps per agent in the first generation (default: 200)
- `END_MAX_STEPS`: Maximum number of steps per agent in the last generation (default: 1000)
- `STEPS_INCREMENT`: Increment of maximum steps per generation (default: 50)

You can adjust these values based on the desired trade-off between exploration and training time.

## How it Works

### Neural Network:
- The agent’s neural network takes the environment’s state as input (a flattened, normalized RGB image of the screen).
- The network's output represents the predicted action, with the agent choosing the action that maximizes the output.
- The network is simple, with weights that are updated through the genetic algorithm.

### Genetic Algorithm:
- **Selection**: Tournament selection is used to choose two parents from the population based on their fitness scores (total reward).
- **Crossover**: A two-point crossover is used to combine the parents' weights and create two offspring.
- **Mutation**: A small mutation (random noise) is applied to the offspring's weights to introduce genetic diversity.
- **Elitism**: The top 30% of agents (elites) are directly carried over to the next generation, ensuring that the best solutions are preserved.

### Fitness Evaluation:
- Each agent is evaluated by interacting with the Pong environment. The total reward (score) obtained during the game is used as the fitness score.
- The agent’s performance is tested for `CURRENT_MAX_STEPS` steps, and its total reward is calculated. The best agent in each generation is saved.

### Video Recording:
- After each generation, the best-performing agent’s gameplay is recorded and saved as a video in the `saved-video-folder/`.
- The video files are named according to the generation number and agent's performance.

### Fitness History:
- A plot of the best agent's total reward over generations is displayed after the training process. This shows how the agent improves its performance over time.

## Example Output

### Console Output:
```
Generation 1/50
Max Steps for Generation 1: 200
Evaluating Agent 1 in Generation 1
Evaluating Agent 2 in Generation 1
.
.
.
Best agent of Generation 1: Fitness Score = 250
.
.
.
```

### Fitness History Plot:
A plot will be generated at the end of the training showing the agent's fitness (total reward) over generations.

### Video:
Gameplay videos of the best agent in each generation will be saved in the `saved-video-folder/`.

## Contributors
### 2201AI01 - Adil
### 2201AI13 - Harpranav
### 2201AI15 - Harshit
### 2201AI22 - Lokesh
### 2201AI48 - Divyam