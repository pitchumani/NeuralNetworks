#ifndef MLP_H_
#define MLP_H_
// Multi-Layer Perceptron (MLP) header file

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

// Perceptron class definition
// This class represents a simple perceptron model that can be used as
// a building block for more complex neural networks.
// This is a model of the neuron which is the basic unit of a neural network.
// The perceptron takes multiple inputs, applies weights to them, and produces a single output.
// The output is typically passed through an activation function to introduce non-linearity.
// bias is an additional parameter that allows the model to fit the data better. it is not
// given as input but is learned during training.
class Perceptron {
public:
    vector<double> weights; // Weights for the inputs
    double bias;            // Bias term for the perceptron
    Perceptron(size_t input_size, double bias = 1.0);
    ~Perceptron() = default;
    double run(vector<double> inputs) const; // Run the perceptron with given inputs
    static double frand();
    void setWeights(const vector<double>& new_weights); // Set the weights of the perceptron
    double sigmoid(double x) const; // Sigmoid activation function
};

class MultiLayerPerceptron {
protected:
    vector<size_t> layers; // Layers of perceptrons
    double bias;          // Bias term for the MLP
    double learning_rate; // Learning rate for weight updates

public:
    MultiLayerPerceptron(vector<size_t> layers, double bias = 1.0, double learning_rate = 0.5);
    ~MultiLayerPerceptron() = default;
    // network of perceptrons
    vector<vector<Perceptron>> Network;
    // output values
    vector<vector<double>> Values;
    // error terms
    vector<vector<double>> Errors;

    // Run the MLP with given inputs
    vector<double> run(const vector<double>& inputs);
    // Set the weights of the MLP
    void setWeights(const vector<vector<vector<double>>>& new_weights);
    // Print the weights of the MLP
    void printWeights() const;
};

#endif // MLP_H_