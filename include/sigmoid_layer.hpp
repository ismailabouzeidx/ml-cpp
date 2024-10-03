#pragma once

#include "layer.hpp"

class sigmoid_layer : public layer{

    public:
        sigmoid_layer(int input, int output);
        std::vector<float> forward (std::vector<float> &input) override;
    private:
        float sigmoid(float x);
};