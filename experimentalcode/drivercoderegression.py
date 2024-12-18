import numpy as np
from DataStructures.linklist import LinkList
from Layers.ActivationLayer import ActivationLayer
from Layers.DenseLayer import DenseLayer
from Layers.InputLayer import InputLayer
from Layers.ActivationFunctions import sigmoid,sigmoid_derivative,relu,relu_derivative

from Layers.lossFunctions import binary_cross_entropy,mse_loss

input_features = np.array([
    [1436.35, 1, 24, 13.42],
    [2876.79, 4, 49, 29.03],
    [2329.98, 5, 22, 28.94],
    [1996.65, 4, 30, 25.74],
    [890.05, 5, 29, 9.54],
    [1500.25, 3, 28, 20.57],
    [2200.60, 2, 35, 18.21],
    [2800.75, 6, 50, 30.48],
    [1800.60, 4, 40, 25.38],
    [2400.80, 5, 37, 27.98],
    [1550.10, 3, 32, 15.62],
    [2900.25, 4, 48, 33.21],
    [2100.50, 5, 40, 22.79],
    [1200.75, 2, 20, 18.03],
    [2500.30, 3, 38, 27.81],
    [2650.60, 4, 42, 30.01],
    [1700.45, 3, 36, 24.57],
    [3000.20, 6, 55, 35.60],
    [2150.35, 4, 44, 26.48],
    [1300.55, 2, 28, 15.34],
    [2400.25, 5, 43, 26.98],
    [1800.90, 3, 29, 23.10],
    [2200.65, 4, 50, 28.57],
    [1950.15, 5, 39, 25.20],
    [2800.90, 6, 51, 32.81],
    [2100.45, 4, 41, 27.33],
    [1600.80, 3, 34, 20.10],
    [2300.75, 5, 47, 28.02],
    [1950.90, 4, 36, 24.72],
    [2700.60, 6, 53, 32.50],
    [2600.40, 5, 46, 31.28],
    [2100.85, 3, 29, 23.90],
    [2200.10, 4, 44, 27.25],
    [2400.20, 6, 50, 30.45],
    [2800.35, 5, 45, 31.40],
    [2300.55, 4, 42, 28.00],
    [2100.75, 3, 34, 25.00],
    [1800.25, 2, 30, 18.88],
    [1500.35, 4, 25, 22.95],
    [2600.30, 5, 40, 29.70],
    [2000.75, 3, 38, 26.18],
    [2700.80, 6, 53, 34.25],
    [2500.20, 4, 41, 28.77],
    [2300.95, 3, 31, 25.11],
    [2100.35, 5, 45, 29.25],
    [2200.75, 4, 40, 26.81],
    [1800.90, 3, 30, 23.40],
    [1700.45, 5, 36, 20.90],
    [2000.25, 4, 37, 27.19],
    [2100.60, 2, 33, 24.52],
    [2900.60, 6, 50, 33.50],
    [2400.35, 4, 42, 28.20],
    [1800.60, 3, 35, 22.90],
    [2500.10, 5, 45, 30.00],
    [2300.90, 4, 43, 26.25],
    [2100.45, 3, 32, 24.80],
    [2700.90, 6, 50, 32.20],
    [2800.35, 5, 48, 31.10],
    [2900.80, 4, 50, 33.00],
    [2000.25, 2, 28, 22.20],
    [2400.60, 5, 44, 29.10],
    [2500.25, 4, 39, 27.50],
    [2100.80, 3, 34, 25.20],
    [2200.95, 4, 45, 28.90],
    [2400.15, 5, 42, 30.10],
    [2200.10, 4, 44, 26.40],
    [1800.90, 3, 30, 21.70],
    [2000.75, 4, 36, 23.50],
    [2300.55, 5, 47, 29.60],
    [2500.40, 6, 50, 31.30],
    [2700.80, 5, 51, 32.10],
    [2000.90, 4, 38, 24.80],
    [2900.70, 6, 53, 34.40],
    [2100.30, 4, 42, 26.80],
    [2400.10, 5, 43, 29.50],
    [2200.60, 3, 30, 23.90],
    [2800.20, 6, 51, 33.20],
    [2600.70, 5, 46, 30.00],
    [2500.15, 4, 44, 28.50],
    [2700.50, 5, 47, 32.30],
    [2000.35, 3, 33, 26.10],
    [2300.25, 4, 42, 27.00],
    [2100.80, 3, 31, 23.80],
    [2000.65, 4, 40, 24.20],
    [2500.10, 5, 44, 30.20],
    [2400.60, 6, 50, 32.10],
    [2300.35, 4, 38, 29.80],
    [2700.20, 5, 49, 33.10],
    [2200.55, 3, 34, 25.50],
    [2800.75, 5, 48, 31.40],
    [2500.45, 4, 42, 28.70],
    [2400.10, 6, 50, 32.80],
    [2200.90, 5, 46, 30.50],
    [2000.60, 4, 39, 26.70],
    [2800.80, 6, 50, 33.40]
    
])

# Prices corresponding to the above input features
price = np.array([
    274138.8, 608990.25, 586635.4, 463824.7, 263833.72,
    320000.0, 470000.5, 620000.1, 500000.0, 550000.25,
    330000.0, 470000.5, 500000.1, 400000.0, 300000.0,
    500000.5, 510000.7, 430000.3, 370000.0, 590000.2,
    420000.1, 340000.0, 430000.5, 310000.0, 450000.8,
    500000.2, 340000.5, 520000.3, 450000.0, 540000.5,
    580000.0, 600000.5, 490000.0, 600000.2, 550000.1,
    510000.6, 570000.3, 530000.4, 560000.0, 520000.0,
    540000.7, 570000.1, 500000.3, 530000.2, 490000.7,
    510000.8, 470000.0, 510000.0, 560000.2, 550000.3,
    530000.6, 470000.4, 590000.1, 600000.6, 580000.5,
    520000.3, 530000.7, 600000.1, 580000.3, 570000.6,
    560000.0, 510000.1, 550000.2, 510000.2, 540000.0,
    480000.0, 500000.4, 540000.6, 550000.1, 530000.8,
    520000.2, 510000.3, 580000.3, 600000.5, 560000.6,
    530000.4, 490000.0, 540000.2, 550000.6, 510000.9,
    600000.3, 500000.1, 530000.0, 570000.4, 560000.2,
    550000.0, 580000.6, 520000.5, 530000.6, 600000.0,
    490000.5, 540000.3, 510000.4, 530000.5, 580000.4,
])

print(input_features.shape)
print(price.shape)

nn = LinkList()

# Example of modifying the architecture for regression (adjusting output layer to a single value)

nn.insert_node(DenseLayer(len(input_features[0]), 64))  # First dense layer
nn.print_previous_output_size()
nn.insert_node(ActivationLayer(relu, relu_derivative))  # Activation function after first layer
nn.print_previous_output_size()

nn.insert_node(DenseLayer(None, 32))  # Second dense layer
nn.print_previous_output_size()
nn.insert_node(ActivationLayer(relu, relu_derivative))  # Activation function after second layer
nn.print_previous_output_size()

nn.insert_node(DenseLayer(None, 8))  # Second dense layer
nn.print_previous_output_size()
nn.insert_node(ActivationLayer(relu, relu_derivative))  # Activation function after second layer
nn.print_previous_output_size()


nn.insert_node(DenseLayer(None, 1))  # Output layer (only 1 neuron for regression)
nn.print_previous_output_size()

# Training loop
learning_rate = 0.01
for i in range(10):
    predicted_output = nn.forward_propogation(input_features)  # Forward pass
    print(predicted_output.shape)
    error = price - predicted_output  # Calculate error
    loss = mse_loss(price, predicted_output)  # Calculate MSE loss
    print(f"Loss at iteration {i}: {loss}")

    gradients_stack = nn.backward_propagation(error)  # Backpropagate error
    nn.update_parameters(gradients_stack, learning_rate)  # Update weights and biases
