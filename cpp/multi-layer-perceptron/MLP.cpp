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

MultiLayerPerceptron::MultiLayerPerceptron(vector<size_t> layers, double bias, double learning_rate)
    : layers(layers), bias(bias), learning_rate(learning_rate) {
    // create neurons layer by layer
    for (size_t i = 0; i < layers.size(); ++i) {
        // Initialize output values for each layer
        Values.push_back(vector<double>(layers[i], 0.0));
         // Initialize the network layer (empty for now)
        Network.push_back(vector<Perceptron>());
        // Network[0] is input layer, so no perceptrons are created
        if (i > 0) {
            for (size_t j = 0; j < layers[i]; ++j) {
                // Create a perceptron for each neuron in the previous layer
                // The input size is the size of the previous layer
                size_t input_size = layers[i - 1];
                Network[i].push_back(Perceptron(input_size, bias));
            }
        }
    }
}

// Set weights for the perceptrons in the each layer in the network,
// except for the input layer which does not have perceptrons.
void MultiLayerPerceptron::setWeights(const vector<vector<vector<double>>>& new_weights) {
    for (size_t layer = 0; layer < new_weights.size(); ++layer) {
        for (size_t perceptron = 0; perceptron < new_weights[layer].size(); ++perceptron) {
            Network[layer + 1][perceptron].setWeights(new_weights[layer][perceptron]);
        }
    }
}

vector<double> MultiLayerPerceptron::run(const vector<double>& inputs) {
    // Set the input values for the first layer
    Values[0] = inputs;

    // Forward pass through the network
    for (size_t i = 1; i < Network.size(); ++i) {
        for (size_t j = 0; j < layers[i]; ++j) {
            Values[i][j] = Network[i][j].run(Values[i - 1]);
        }
    }

    // Return the output of the last layer
    return Values.back();
}

void MultiLayerPerceptron::printWeights() const {
    // start from the second layer (skipped the input layer)
    cout << "Weights of the Multi-Layer Perceptron:" << endl;
    for (size_t i = 1; i < Network.size(); ++i) {
        for (size_t j = 0; j < layers[i]; ++j) {
            cout << "Layer " << i + 1 << " Neuron " << j << " weights: ";
            for (const auto& weight : Network[i][j].weights) {
                cout << weight << " ";
            }
            cout << endl;
        }
    }
    cout << endl;
}
