#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>   
#include <random>  

#include "layer.hpp"

class NN{
    public:
        NN(float lr);
        ~NN();
        
        void add_layer(layer* layer);
        void forward(std::vector<float> &input);
        void init_weights();
        void print_neurons();

    private:
        float learning_rate;

        std::vector<layer*> layers;
};