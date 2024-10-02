#include "layer.hpp"

class fully_connected_layer : public layer {

    public:
        fully_connected_layer(int input, int output);

        void forward(std::vector<float> &inputs) override;
};