# simple_ann
A simple Artificial Neural Network architecture built from scratch.

## Problem Type :grey_question:

We have taken up the problem for Multi-class Classification.

## Model Choice :white_check_mark:

As the repo name suggests, simple ANN ( Artificial Neural Network ) is what we have chosen for solving the use-case described below.

## Input Data :1234:

We are trying to classify an image of handwritten digit ( 0 - 9 ) belonging to the [MNIST database](https://web.archive.org/web/20200430193701/http://yann.lecun.com/exdb/mnist/).

## Artificial Neural Network ( ANN ) :book:

We choose the following architecture for our neural network : 
```
Layer 1 : Input Layer ( 28 * 28 ) is flattened  
Layer 2 : Hidden Layer ( 128 neurons )
Layer 3 : Hidden Layer ( 128 neurons )
Layer 4 : Output Layer ( 10 neurons )
```
The first 3 layers make use of a ReLU function as the activation function. The last layer makes use of a Softmax Function as the activation function.

## Tech Stack :computer:
```
1. Python
2. Tensorflow
3. Keras
```

## Skills :mortar_board:
```
1. Python Programming
2. Deep Learning
3. Artificial Neural Networks (ANN)
4. Model Building from Scratch using Tensorflow/Keras libraries
5. GPU Acceleration
```

## Pre-Requisites :white_check_mark:
```
1. Python 3.10
2. 
```

## Get Started :rocket:

To get started simply take the steps below :
```
git clone 
cd simple_ann
python -m venv .venv
.venv\Scripts\activate #If on windows, macOS requires : source .venv/bin/activate
pip install tensorflow==2.10.1
cd training
python model_train.py --batch_size <> --epochs <> --learning_rate <>
```
Replace <> above with your own values. Sample values:
```
Batch Size : 64
Epochs : 100
Learning Rate : 0.05
``` 

## Results :white_check_mark:

The training conducted with the above sample values provide an accuracy of around 97.71%.

## Author :email:

Yash Mahesh Bangera <br>
Email : yashmaheshbangera.work@gmail.com