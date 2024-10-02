#include "fully_connected_layer.hpp"

fully_connected_layer::fully_connected_layer(int input, int output){
    std::cout << "Constructed a fully connected layer with input: " << input <<" & output: " << output << std::endl;
    
    this->input_size = input;
    this->output_size = output;

    this->neurons = std::vector<float>(this->output_size);
    this->weights = std::vector<std::vector<float>>(this->output_size,std::vector<float>(this->input_size));
}

void fully_connected_layer::forward(std::vector<float> &input){

    for (int i =0 ; i< this->output_size; i++){
        this->neurons[i] = 0.0f;
        for (int j =0 ; j < this->input_size; j++) {

            this->neurons[i] += input[j] * this->weights[i][j];
        }
    }
}