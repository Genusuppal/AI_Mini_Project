import numpy as np
import random

MUTATION_RATE = 0.1  # Lower mutation rate
MUTATION_SCALE = 0.05  # Smaller mutation scale

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        
    def forward(self, x):
        return np.dot(x, self.weights)  # Simple linear model for action prediction
        
    def mutate(self):
        # Mutation: randomly add noise to weights
        if random.random() < MUTATION_RATE:
            self.weights += np.random.randn(*self.weights.shape) * MUTATION_SCALE

    def copy(self):
        # Return a copy of the current network
        new_nn = NeuralNetwork(self.weights.shape[0], self.weights.shape[1])
        new_nn.weights = self.weights.copy()
        return new_nn

    def save_weights(self, file_path):
        np.save(file_path, self.weights)

    def load_weights(self, file_path):
        self.weights = np.load(file_path)