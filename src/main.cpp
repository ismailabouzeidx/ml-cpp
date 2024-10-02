#include <iostream>
#include "NN.hpp"
#include "fully_connected_layer.hpp"

int main() {
    NN net(0.01f);
    
    net.add_layer(new fully_connected_layer(2,4));
    net.add_layer(new fully_connected_layer(4,1));

    net.init_weights();

    std::vector<float> in = {1.0 , 1.0};
    net.forward(in);


    net.print_neurons();


    return 0;
}
