# ml-cpp

**ml-cpp** is a machine learning library written in C++ designed for ease of use, flexibility, and performance. The library aims to provide fundamental machine learning algorithms and tools to help users understand and implement machine learning concepts efficiently.

## Current Features

- **Fully Connected Neural Network (FCNN)**: 
  - Implementation of fully connected layers, allowing users to create deep learning models.
  
- **Activation Functions**:
  - Sigmoid activation function for non-linear transformations in neural networks.

- **Loss Functions**:
  - Mean Squared Error (MSE) loss function for regression tasks.

- **Backpropagation**:
  - Complete implementation of backpropagation for training neural networks.

- **Training Routine**:
  - Basic training routine that supports feeding input and target data, adjusting weights based on the computed gradients.

- **Prediction**:
  - Capability to make predictions after training the model on provided datasets.

## Upcoming Features

- **Shared Libs**:
  - Implement support for building shared libraries to allow users to link the library dynamically with their applications. This will help reduce memory usage and enable easier updates to the library without recompiling dependent applications.

- **Additional Activation Functions**:
  - Implement more activation functions like ReLU, Leaky ReLU, and Softmax to enhance model performance.

- **Convolutional Neural Networks (CNN)**:
  - Support for convolutional layers to facilitate image processing and computer vision tasks.

- **Optimization Algorithms**:
  - Incorporate advanced optimization techniques such as Adam, RMSprop, and SGD with momentum.

- **Regularization Techniques**:
  - Implement dropout and L2 regularization to prevent overfitting during training.

- **Data Preprocessing**:
  - Functions for normalizing and preprocessing datasets before feeding them into the model.

- **Model Saving and Loading**:
  - Functionality to save trained models to disk and load them back for inference.

- **Unit Tests**:
  - A suite of unit tests to ensure the correctness and reliability of the implemented algorithms.

## Installation

To build and run the **ml-cpp** project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ismailabouzeidx/ml-cpp.git
   cd ml-cpp
   ```

2. Compile the project using CMake
    ```bash
    mkdir build && cd build
    cmake ..
    make -j
    ```
3. Run the program 
    ```bash
    ./ml-cpp
    ```
4. Example Usage
    ```cpp
    // Example usage
    NN net(0.01f);
    net.add_layer(std::make_unique<fully_connected_layer>(2, 4));
    net.add_layer(std::make_unique<sigmoid_layer>(4, 4));
    net.add_layer(std::make_unique<fully_connected_layer>(4, 1));
    net.add_layer(std::make_unique<sigmoid_layer>(1, 1));

    // Train the network with your input and target data
    net.train(input_data, target_data, epochs);
    ```

