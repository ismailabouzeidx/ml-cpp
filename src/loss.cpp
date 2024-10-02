#include "loss.hpp"

float mse_loss(const std::vector<float>& target, const std::vector<float>& prediction) {
    float loss = 0.0f;
    for (size_t i = 0; i < target.size(); ++i) {
        loss += std::pow(target[i] - prediction[i], 2);
    }
    return loss / target.size();
}

std::vector<float> mse_loss_derivative(const std::vector<float>& target, const std::vector<float>& prediction) {
    std::vector<float> grad(prediction.size());
    for (size_t i = 0; i < prediction.size(); ++i) {
        grad[i] = 2 * (prediction[i] - target[i]) / target.size();
    }
    return grad;
}