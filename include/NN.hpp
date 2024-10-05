#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>   
#include <random>  

#include "layer.hpp"
#include "loss.hpp"

class NN{
    public:
        NN(float lr);
        ~NN();
        
        void add_layer(std::unique_ptr<layer> layer);
        void print_neurons();
        void init_weights();
        
        std::vector<float> forward(std::vector<float> &input);
        void backward(std::vector<float> &target);

        void train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, int epochs);
        
        std::vector<float> predict(std::vector<float> &input);

    private:
        float learning_rate;

        std::vector<std::unique_ptr<layer>> layers;
};