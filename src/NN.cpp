#include "NN.hpp"

NN::NN(float lr){
    this->learning_rate = lr;
}

NN::~NN(){}

void NN::add_layer(std::unique_ptr<layer> layer){
    layer->learning_rate = this->learning_rate;
    this->layers.emplace_back(std::move(layer));
}

void NN::init_weights(){
    std::random_device rd;  // Seed for random number generator
    std::mt19937 gen(rd()); // Random number generator

    for (const auto& layer : this->layers) {
        // Calculate the Xavier initialization limit based on layer sizes
        float limit = sqrt(6.0f / (layer->input_size + layer->output_size));
        std::uniform_real_distribution<> dis(-limit, limit); // Range [-limit, limit]

        // Initialize weights for this layer
        for (int i = 0; i < layer->output_size; ++i) {
            for (int j = 0; j < layer->input_size; ++j) {
                layer->weights[i][j] = dis(gen);  // Assign random weight between -limit and limit
            }
        }
    }
}

void NN::print_neurons(){
    for (const auto &layer : this->layers){
        for (auto &out : layer->neurons){
            std::cout << "Out: " << out << std::endl;
        }
    }
}

std::vector<float> NN::forward(std::vector<float> &input){
    auto current_input = input;
    for (const auto &layer : this->layers){
        current_input = layer->forward(current_input);
    }
    return current_input;
}

void NN::backward(std::vector<float>& target) {
    // Calculate initial error at the output layer (direct error)
     std::vector<float> dL_dO = mse_loss_derivative(target, this->layers.back()->neurons);

    // Propagate the error backward through each layer
    for (int i = this->layers.size() - 1; i >= 0; --i) {
        dL_dO = this->layers[i]->backward(dL_dO);
    }
}


void NN::train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, int epochs) {
    this->init_weights();

    // Step 2: Training loop for specified epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<float> input = inputs[i];
            std::vector<float> target = targets[i];

            // Forward pass for each input-target pair
            std::vector<float> out = this->forward(input);

            // Calculate the loss
            float loss = mse_loss(target, out);
            total_loss += loss;

            // Backward pass
            this->backward(target);
        }

        // Output the loss for the current epoch
        std::cout << "Epoch: " << epoch + 1 << "/" << epochs << ", Loss: " << total_loss / inputs.size() << "\n";
    }
}


std::vector<float> NN::predict(std::vector<float> &input) {
    // Perform forward pass
    this->forward(input);
    
    // Return the output from the last layer (predicted values)
    return this->layers.back()->neurons;
}

