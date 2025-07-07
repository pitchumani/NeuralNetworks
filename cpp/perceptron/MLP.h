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

// Perceptron is used for linear classification tasks (classification). As it can deduce the
// linearly separable data, it is a fundamental building block for more complex neural networks.
// It can be used to implement simple logic gates like AND, OR, and NOT.
// The XOR gate, however, is not linearly separable and requires a more complex network to model it.
// The single perceptron can only learn linearly separable functions. We need a multi-layer perceptron (MLP)
// to learn non-linear functions like XOR.
// NAND and OR followed by a AND gate can be used to implement XOR.

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

#endif // MLP_H_