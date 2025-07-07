#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <cmath>

// This is a simple implementation of a perceptron model in C++
// A perceptron is a type of artificial neuron that takes multiple inputs, applies weights to them,
// and produces a single output. It is the basic building block of neural networks.

#include "MLP.h"

int main() {
    srand(time(NULL)); // Seed the random number generator
    rand(); // Generate a random number to ensure randomness

    std::cout << "Neural Networks Examples" << std::endl;
    std::cout << "== AND Gate Example ==" << std::endl;
    Perceptron and_gate(2); // Create a perceptron with 2 inputs
    // select weights to inputs and bias for AND gate
    // The weights are chosen such that the perceptron outputs 1 only when both inputs are 1.
    and_gate.setWeights({10, 10, -15}); // set weights to inputs and bias for AND gate
    std::cout << "AND(0, 0) = " << and_gate.run({0, 0}) << std::endl;
    std::cout << "AND(0, 1) = " << and_gate.run({0, 1}) << std::endl;
    std::cout << "AND(1, 0) = " << and_gate.run({1, 0}) << std::endl;
    std::cout << "AND(1, 1) = " << and_gate.run({1, 1}) << std::endl;

    std::cout << "== OR Gate Example ==" << std::endl;
    Perceptron or_gate(2); // Create a perceptron with 2 inputs
    // select weights to inputs and bias for OR gate
    // The weights are chosen such that the perceptron outputs 1 when at least one input is 1.
    // The bias is set to a value that allows the perceptron to output 1 for the OR condition.
    or_gate.setWeights({15, 15, -10}); // set weights to inputs and bias for OR gate
    std::cout << "OR(0, 0) = " << or_gate.run({0, 0}) << std::endl;
    std::cout << "OR(0, 1) = " << or_gate.run({0, 1}) << std::endl;
    std::cout << "OR(1, 0) = " << or_gate.run({1, 0}) << std::endl;
    std::cout << "OR(1, 1) = " << or_gate.run({1, 1}) << std::endl;

    return 0;
}