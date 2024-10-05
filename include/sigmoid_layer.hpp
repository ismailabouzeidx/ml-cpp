#pragma once

#include "layer.hpp"

class sigmoid_layer : public layer{

    public:
        sigmoid_layer(int input, int output);
        std::vector<float> forward(std::vector<float> &inputs) override;
        std::vector<float> backward(std::vector<float>& dL_dO) override;
    private:
        float sigmoid(float x);
        float sigmoid_derivative(float x);
};