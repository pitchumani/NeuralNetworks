# Neural Networks in C++
A collection of neural network implementations in C++.

## Examples

- [Perceptron](./perceptron): Demonstrates how to build a simple single-layer perceptron, the most basic form of a neural network.
- [Multi-Layer Perceptron](./multi-layer-perceptron): Shows how to implement a multi-layer perceptron for more complex tasks.

Implemented these examples following the [Training Neural Networks in C++ course](https://www.linkedin.com/learning/training-neural-networks-in-c-plus-plus-22661958) in LinkedIn Learning.

The examples are made to behave like AND, OR and XOR gates by providing the exact weights the model needed.
The real ability of Neural Networks is to learn. If it learns from the examples of how an XOR gate behaves, then it would be good.

Dataset:
It is a collection of samples with features and labels. Features are input data, where labels are the known categories of each sample.
single training sample:
- feed the an input X to network
- get the output and compare it with the correct value Y
- calculate the error
- use this error to adjust the weights

