#include "MLP.h"

double Perceptron::frand() {
    // generate a random number in the range [0, 1]
    // scaling factor is 2.0 and shift of -1.0
    return (2.0 * (double)rand() / RAND_MAX) - 1.0; // Generate a random number in the range [-1, 1]
}

Perceptron::Perceptron(size_t input_size, double bias) : bias(bias) {
    weights.resize(input_size + 1); // +1 for the bias weight
    generate(weights.begin(), weights.end(), frand);
    for (size_t i = 0; i < input_size; ++i) {
        weights[i] = 0.0; // Initialize weights to zero
    }
}

double Perceptron::run(vector<double> inputs) const {
    inputs.push_back(bias); // Add the bias term to the inputs
    double sum = inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0);
    return sigmoid(sum); // Apply the sigmoid activation function
}

void Perceptron::setWeights(const vector<double>& new_weights) {
    if (new_weights.size() != weights.size()) {
        cerr << "Error: Weight size mismatch." << endl;
        return;
    }
    weights = new_weights; // Set the weights to the new values
}

// Sigmoid activation function
// This function takes a real-valued input and returns a value between 0 and 1.
double Perceptron::sigmoid(double x) const {
    // 1 / (1 + e^-x) where e is euler's number (approximately 2.71828)
    return 1.0 / (1.0 + exp(-x));
}