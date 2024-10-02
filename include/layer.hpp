#pragma once

#include <iostream>
#include <vector>

class layer{
    
    public:
        layer();
        
        virtual void forward(std::vector<float> &input) = 0;
    
    public:
        std::vector<std::vector<float>> weights;
        std::vector<float> neurons;

        int input_size;
        int output_size;
};