#include "activations.hpp"

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

// Derivative of Sigmoid for backprop
float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}