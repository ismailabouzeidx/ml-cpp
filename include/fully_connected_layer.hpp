#include "layer.hpp"
#include "activations.hpp"

class fully_connected_layer : public layer {

    public:
        fully_connected_layer(int input, int output);

        std::vector<float> forward(std::vector<float> &inputs) override;
};