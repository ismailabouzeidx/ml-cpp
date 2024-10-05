#pragma once

#include <iostream>
#include <vector>
#include <cmath>

class layer{
    
    public:
        layer();
        
        virtual std::vector<float> forward(std::vector<float> &inputs) = 0;
        virtual std::vector<float> backward(std::vector<float>& dL_dO) = 0;
        
    
    public:
        std::vector<std::vector<float>> weights;
        std::vector<float> neurons;
        std::vector<float> inputs;

        int input_size;
        int output_size;

        float learning_rate;
};