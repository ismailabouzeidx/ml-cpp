#pragma once

#include <vector>
#include <cmath>

float mse_loss(const std::vector<float>& target, const std::vector<float>& prediction);

std::vector<float> mse_loss_derivative(const std::vector<float>& target, const std::vector<float>& prediction);