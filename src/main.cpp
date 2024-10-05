#include <iostream>
#include "NN.hpp"
#include "fully_connected_layer.hpp"
#include "sigmoid_layer.hpp"

int main() {
    // Create a neural network with learning rate 0.01
    NN net(0.1f);

    // Add layers to the network
    net.add_layer(std::make_unique<fully_connected_layer>(2, 4));  // Fully connected input layer (2 input, 4 output neurons)
    net.add_layer(std::make_unique<sigmoid_layer>(4, 4));          // Sigmoid activation for the hidden layer
    net.add_layer(std::make_unique<fully_connected_layer>(4, 1));  // Output layer (4 input neurons, 1 output neuron)
    net.add_layer(std::make_unique<sigmoid_layer>(1, 1));          // Sigmoid activation for the output layer

    // XOR inputs and targets (full dataset)
    std::vector<std::vector<float>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<float>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Train the network for 30000 epochs
    int epochs = 30000;
    net.train(inputs, targets, epochs);

    // Test the network after training
    for (auto& test_input : inputs) {
        std::vector<float> predictions = net.predict(test_input);
        std::cout << "Input: (" << test_input[0] << ", " << test_input[1] << ") -> Prediction: " << predictions[0] << std::endl;
    }

    return 0;
}
