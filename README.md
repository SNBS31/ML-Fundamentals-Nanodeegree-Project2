# Handwritten Digit Recognition with MNIST

## Project Overview

This project is part of the AWS Machine Learning Fundamentals Nanodegree program. The goal is to prototype a system for optical character recognition (OCR) using the MNIST database of handwritten digits. The project involves preprocessing the dataset, building a neural network, and training and tuning the model to achieve high accuracy in digit classification.

## Table of Contents

- [Project Summary](#project-summary)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Instructions](#instructions)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Summary

In this project, I performed the following steps:

1. **Data Loading and Exploration**: Loaded the MNIST dataset, transformed the data into tensors, and created DataLoader objects for training and testing.
2. **Model Design and Training**: Built a neural network using PyTorch with at least two hidden layers, selected an appropriate loss function, and defined an optimizer to minimize the loss.
3. **Model Testing and Evaluation**: Evaluated the model's accuracy on the test set, tuned hyperparameters, and achieved at least 90% classification accuracy.
4. **Model Saving**: Saved the trained model parameters using `torch.save()` for future use.

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Dataset

The dataset used for this project is the MNIST database, which consists of 70,000 images of handwritten digits (0-9). The dataset is made available through `torchvision.datasets`.

## Instructions I had to follow, in order to successfully pass Udacity's project review

1. **Load the Dataset**:
   - Utilize `torchvision.datasets` to load the MNIST dataset.
   - Apply transformations to convert the data to tensors, normalize, and flatten the images.

2. **Visualize the Dataset**:
   - Use provided functions to visualize the dataset.
   - Explore the size and shape of the data to understand the inputs.

3. **Build the Neural Network**:
   - Create a neural network architecture using PyTorch.
   - Implement an optimizer to update the network's weights.

4. **Train the Model**:
   - Use the training DataLoader to train the neural network.

5. **Evaluate the Model**:
   - Test the model's accuracy on the test set.
   - Tune hyperparameters and network architecture to achieve at least 90% accuracy.

6. **Save the Model**:
   - Use `torch.save()` to save the trained model parameters.

## Model Architecture

The neural network consists of:
- Input layer
- Three hidden layers with activation functions
- Output layer with softmax activation to predict probabilities for each of the 10 classes.
Here's a breakdown:
- Input Layer: Flattened image data (size 784) - (Implicit)
- Hidden Layer 1: fc1 with 128 neurons
- Hidden Layer 2: fc2 with 90 neurons
- Hidden Layer 3: fc3 with 50 neurons
- Output Layer (Optional): fc4 with 10 neurons (if added with softmax activation)

## Results

The performance that I got was an overall testing accuracy of 97.26%.
Such a score is considered quite good, since achieving high accuracy on the MNIST dataset is a challenging task as it requires the model to correctly classify handwritten digits with high precision.

## Conclusion

This project successfully demonstrates the application of deep learning techniques for image classification using PyTorch. The trained model can be used for further applications in optical character recognition and similar tasks. 

---

Feel free to reach out if you have any questions or would like to discuss the project further!
