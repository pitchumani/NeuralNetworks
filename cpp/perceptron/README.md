# Perceptron Model Implementation in C++

This is a simple implementation of a perceptron model in C++. A perceptron is a type of artificial neuron, which is a computational model inspired by the way biological neurons work. It takes multiple inputs, applies weights to them, and produces a single output. It is the basic building block of neural networks. [Learn more about artificial neurons](https://en.wikipedia.org/wiki/Artificial_neuron).

Perceptron is used for linear classification tasks. As it can deduce the linearly separable data, it is a fundamental building block for more complex neural networks. For example, a perceptron can classify points on a 2D plane into two categories separated by a straight line. It can also be used to implement simple logic gates like AND, OR, and NOT. The XOR gate, however, is not linearly separable and requires a more complex network to model it.

The AND and OR gates are linearly separable because their outputs can be separated by a straight line (or hyperplane) in the input space. For example, in a 2D plot of inputs (A, B), you can draw a line that separates the output 0s from the output 1s for both AND and OR gates. This means a single perceptron can learn these functions.

However, the XOR gate is not linearly separable. There is no straight line that can separate the output 1s from the output 0s for all possible input combinations of XOR. Below is the truth table for XOR:

The single perceptron can only learn linearly separable functions. We need a multi-layer perceptron (MLP), which is a type of neural network with one or more hidden layers that can model non-linear relationships, to learn non-linear functions like XOR. [Learn more about MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron). XOR can be implemented using NAND, OR, and AND gates as follows:
| Input A | Input B | Output XOR |
|---------|---------|------------|
| 0       | 0       | 0          |
| 0       | 1       | 1          |
| 1       | 0       | 1          |
| 1       | 1       | 0          |

Graphically, the points (0,1) and (1,0) (output 1s) cannot be separated from (0,0) and (1,1) (output 0s) by a single straight line. [Learn more about XOR and its graphical representation](https://en.wikipedia.org/wiki/XOR_gate). This is why a single perceptron cannot learn the XOR function; it requires a network with at least one hidden layer to capture the non-linear relationship.

The XOR gate outputs `1` only when the inputs differ. This behavior can be implemented using a combination of NAND, OR, and AND gates. Specifically, the XOR function can be expressed as:

1. Compute NAND of the inputs: `NAND = NOT(A AND B)`
2. Compute OR of the inputs: `OR = A OR B`
3. Compute AND of the results: `XOR = NAND AND OR`
