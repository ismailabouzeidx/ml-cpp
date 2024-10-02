#include "NN.hpp"

NN::NN(float lr){
    this->learning_rate = lr;
}
NN::~NN(){}
void NN::add_layer(std::unique_ptr<layer> layer){
    this->layers.emplace_back(std::move(layer));
}

void NN::init_weights(){
    std::random_device rd;  // Seed for random number generator
    std::mt19937 gen(rd()); // Random number generator

    for (const auto& layer : this->layers) {
        std::cout << "LAYER\n";
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
void NN::forward(std::vector<float> &input){
    for (const auto &layer : this->layers){
        layer->forward(input);

    }
}