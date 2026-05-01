# ML concepts

Keras Functional API
https://machinelearningmastery.com/keras-functional-api-deep-learning/

## Stat conspets

### Sigmoid

### TanH

### ReLU

## CNN

## RNN

## GAN

## LSTM

* its always good to have next layer as Dropout which ensure no overfitting happens

## K-Fold

* used for cross validation technique
* decides how many folds (partitions) of the data, and 1 part is used for validation, other for test. and every iteration uses a next part/fold for validation


## Hyperparameter

* Filter size (kernel size) 
    * learned value
        * Standard size of 3*3 OR 5*5
        * can be any random value, using Glorot/Xavier initialization
        * in training, backpropagation is used to update the filter weights
    * fixed values
        * Sobel filters
        * Gaussian filters
* Stride

* Padding

* Pooling size  
* Number of filters
* Number of layers  
* Number of epochs

