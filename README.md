# C++ From-Scratch Neural Network (MNIST Classifier)

A lightweight, object-oriented Feedforward Neural Network implemented completely from scratch in C++. This project bypasses high-level machine learning libraries to handle raw matrix calculations, forward propagation, and backpropagation manually using standard library components.

It includes a built-in CSV parser designed to train and evaluate the model using the classic **MNIST handwritten digit dataset**.

## Features

* **Custom Matrix Engine:** Built from scratch supporting matrix multiplication, addition, scalar scaling, Hadamard product, and transpositions.
* **Object-Oriented Architecture:** Flexible `Layer`, `DenseLayer`, and `NeuralNetwork` classes abstraction.
* **Backpropagation & Gradient Descent:** Manual implementation of the chain rule to update weights and biases using a configurable learning rate.
* **Modular Activation Functions:** Uses function pointers to pass normalization functions (e.g., Sigmoid, ReLU).
* **Custom Dataset Parser:** Direct streaming and normalization of MNIST CSV pixel data into matrix column vectors.

## Network Architecture (Default Configuration)

* **Input Layer:** 784 neurons (corresponding to $28 \times 28$ pixel images normalized to a $[0.0, 1.0]$ scale).
* **Hidden Layer:** 64 neurons utilizing a Sigmoid activation function.
* **Output Layer:** 10 neurons representing digits 0–9 via One-Hot encoding.
* **Loss Function:** Mean Squared Error ($MSE$).
