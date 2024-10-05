#include "fully_connected_layer.hpp"

fully_connected_layer::fully_connected_layer(int input, int output){
    std::cout << "Constructed a fully connected layer with input: " << input <<" & output: " << output << std::endl;
    
    this->input_size = input;
    this->output_size = output;

    this->neurons = std::vector<float>(this->output_size);
    this->weights = std::vector<std::vector<float>>(this->output_size,std::vector<float>(this->input_size));
}

std::vector<float> fully_connected_layer::forward(std::vector<float> &inputs){
    this->inputs = inputs;

    for (int i =0 ; i < this->output_size; i++){
        this->neurons[i] = 0.0f;
        for (int j =0 ; j < this->input_size; j++) {
            this->neurons[i] += inputs[j] * this->weights[i][j];
        }
    }
    return this->neurons;
}
std::vector<float> fully_connected_layer::backward(std::vector<float>& dL_dO) {
    // Initialize gradients
    std::vector<float> dL_dI(this->input_size, 0.0f);  // Gradient w.r.t. input to this layer (to return)
    
    // Compute gradients w.r.t. weights and biases, and gradients to propagate back (dL_dI)
    for (int i = 0; i < this->output_size; i++) {
        // Gradient w.r.t bias (same as dL_dO)
        
        for (int j = 0; j < this->input_size; j++) {
            // Gradient w.r.t. weights: dL_dW = dL_dO * inputs[j]
            this->weights[i][j] -= this->learning_rate * dL_dO[i] * this->inputs[j];  // Updating weights in-place
            
            // Gradient w.r.t input: dL_dI = W^T * dL_dO
            dL_dI[j] += this->weights[i][j] * dL_dO[i];  // Accumulate for each input
        }
    }

    return dL_dI;  // Return gradients w.r.t input for further backpropagation
}
