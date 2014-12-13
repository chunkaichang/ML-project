ML-project
===============
## Goal
  Classification of galaxial images using NN, CNN, and SVM

## Image Preprocessing
Each image goes through a 4-step preprocessing before feeding into classification networks.
Step1: Cropping - take the central part
Step2: RGB to GrayScale
Step3: Downsampling
Step4: Whitening

## Neural Networks
3-layer NN is used (2 hidden layers).

Traning Procedure:
  * Input shuffling
  * Forward-backward propagation
  * Gradient Descent
  * Early-stopping checking
  * Learning rate decreasing
 
## Convolutional Neural Networks
  * Architecture: 2 convolutional layers , 2 subsampling layers, 2 fully-connected layers
  * Max-pooling
  
## Support Vector Machines
LIBSVM tool
  
