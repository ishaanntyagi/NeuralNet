# Neural Network Implementation
This repository contains a series of Jupyter notebooks documenting the foundational concepts and step-by-step implementation of Artificial Neural Networks (ANNs) from first principles, primarily using PyTorch (and some TensorFlow for comparison). The goal is to build a solid understanding of how neural networks work under the hood before diving into more complex deep learning architectures.

Table of Contents
01-TensorOps.ipynb

02-PTorch.ipynb

03-ActivationFn.ipynb

04-DataProcessNN.ipynb

05-Perceptron.ipynb

06-perceptron-Learning.ipynb

07-LossFunc.ipynb

08-HiddenLayers.ipynb

09-AdamOptm.ipynb

10-BatchNorm.ipynb

01-TensorOps.ipynb





This notebook introduces fundamental tensor operations, which are the building blocks of neural networks. It covers:

Creating tensors in both TensorFlow and PyTorch.

Performing element-wise operations (e.g., addition).

Executing matrix multiplication.

Demonstrating basic tensor manipulations like reshaping and summing.

02-PTorch.ipynb
A focused introduction to PyTorch tensors. This notebook explores:

Basic tensor creation in PyTorch.

Checking tensor properties such as shape and data type.

03-ActivationFn.ipynb
This notebook delves into various activation functions critical for neural networks. It demonstrates the application of:

ReLU (Rectified Linear Unit)

Sigmoid

Tanh (Hyperbolic Tangent)
Implementations are shown using both torch.nn and tf.nn modules for comparative understanding.

04-DataProcessNN.ipynb
Focuses on the essential steps of preparing data for neural network training, including:

Loading a dataset (e.g., Iris dataset from sklearn).

Splitting data into training and testing sets.

Applying data scaling (e.g., StandardScaler).

Converting NumPy arrays to PyTorch tensors.

Introduction to PyTorch's Dataset and DataLoader for efficient data handling and mini-batching.

05-Perceptron.ipynb
This notebook implements the fundamental unit of a neural network: the Perceptron. It covers:

Defining a Perceptron class from scratch.

Initializing random weights and biases for the perceptron.

Implementing the forward pass to calculate the weighted sum and apply a step-like activation (binary output).

Demonstrating a simple prediction with the initialized perceptron.

06-perceptron-Learning.ipynb
Extends the Perceptron implementation to include a basic learning mechanism. This notebook shows:

How a single perceptron learns from data.

A custom trainingLoop method for the perceptron.

Weight and bias updates based on the error (Perceptron Learning Algorithm).

Demonstration using a simple dataset (e.g., XOR-like logic or a linearly separable dataset).

07-LossFunc.ipynb
Explores the concept and implementation of loss (or cost) functions, which quantify the error of a model's predictions. This notebook demonstrates:

Mean Squared Error (MSE) for regression tasks.

Binary Cross-Entropy (BCE) for binary classification tasks.

Conceptual explanations of different loss functions and their purposes.

08-HiddenLayers.ipynb
Moves beyond a single perceptron to implementing a full Multi-Layer Perceptron (MLP) with hidden layers. This notebook showcases:

Defining a SimpleNeuralNetworkHidden class to build an ANN with multiple hidden layers.

Implementing the full forward pass through these layers, incorporating activation functions.

Setting up trainable parameters with requires_grad=True for automatic differentiation in PyTorch.

Manual implementation of the Backpropagation algorithm and Gradient Descent for updating weights and biases.

Demonstrating the training process and observing the loss reduction over epochs.

09-AdamOptm.ipynb
Introduces a more advanced optimization algorithm, Adam, to improve the training efficiency of neural networks. This notebook:

Refactors the neural network class to leverage torch.nn.Module for better parameter management.

Integrates PyTorch's built-in torch.optim.Adam optimizer.

Compares the training loop using Adam to manual gradient updates.

Demonstrates how Adam automatically handles gradient updates and momentum.

10-BatchNorm.ipynb
Explores the concept and application of Batch Normalization, a technique used to stabilize and accelerate neural network training. This notebook covers:

Introduction to the purpose of Batch Normalization (normalizing layer inputs).

Implementation of nn.BatchNorm1d in a neural network architecture.

Demonstrating the effect of Batch Normalization on input distributions (mean and standard deviation).

How Batch Normalization can lead to faster and more stable convergence.

How to Use
Each notebook is designed to be self-contained and runnable. To explore the code and concepts:

Clone this repository (if applicable) or download the notebooks.

Requirements: Ensure you have Python 3.x installed (preferably 3.9+).

Install the required libraries using pip:

pip install torch tensorflow pandas numpy scikit-learn matplotlib

Running the Notebooks: Open the .ipynb files using Jupyter Notebook or JupyterLab. You can run cells sequentially to follow the explanations and observe the code in action.
