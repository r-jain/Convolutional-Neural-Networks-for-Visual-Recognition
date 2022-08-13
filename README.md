# Convolutional Neural Networks for Visual Recognition

This is the overview of implementations for [**CS231n : Convolutional Neural Networks for Visual Recognition**](http://cs231n.stanford.edu/2019/) by Stanford University.

## Resources

&nbsp;&nbsp;&nbsp;&nbsp;Stanford CS231n Lectures [Playlist.](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

&nbsp;&nbsp;&nbsp;&nbsp;Source code as a zip file:

- [Assignment1](cs231n.github.io/assignments/2019/spring1819_assignment1.zip)
- [Assignment2](cs231n.github.io/assignments/2019/spring1819_assignment2.zip)
- [Assignment3](cs231n.github.io/assignments/2019/spring1819_assignment3.zip)

&nbsp;&nbsp;&nbsp;&nbsp;Environment Setup [Instructions.](https://cs231n.github.io/setup-instructions/)

## Implementations (Author: Rohit Jain)

`python` `numpy` `matplotlib` `jupyter`

> HTML output of jupyter notebooks for my implementattion of the year 2019 version of CS231n is avaibable [here](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/2019/html) for quick reference.
>
> Implementation was done with the following setup, various execution times in the output are from this setup:
>
> - *Processor* - Intel Core 2 Duo CPU P8600 @2.40 GHz
> - *RAM* - 4 GB
> - *OS* - Windows 7
>
>
> Code for my implementations is not posted on this public repository.

### **Assignment 1:**

- Q1: [k-Nearest Neighbor classifier](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/2019/html/assignment1/knn.html)

    `load and visualize CIFAR-10 dataset, subsample test data & train data, compute visualize & vectorize distance matrix, implement k-Nearest Neighbor classifier, Perform k-fold cross validation to find the best value of k, retrain and retest the classifier for the best value of k.`

- Q2: [Training a Support Vector Machine](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/2019/html/assignment1/svm.html)

    `CIFAR-10 data loading and preprocessing, prepare train val and test sets, implement a fully-vectorized loss function for the SVM, implement the fully-vectorized expression for its analytic gradient, check your implementation using numerical gradient, use a validation set to tune the learning rate and regularization strength, optimize the loss function with SGD, visualize the final learned weights.`


- Q3: [Implement a Softmax classifier](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/2019/html/assignment1/softmax.html)

    `CIFAR-10 data loading and preprocessing, prepare train val dev and test sets, implement a fully-vectorized loss function for the Softmax classifier, implement the fully-vectorized expression for its analytic gradient, check your implementation with numerical gradient, use a validation set to tune the learning rate and regularization strength, optimize the loss function with SGD, visualize the final learned weights.`

- Q4: [Two-Layer Neural Network](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/2019/html/assignment1/two_layer_net.html)

    `implement forward pass to compute scores & data and regularization loss, implement backward pass to compute gradient of loss and perform numeric gradient checking, train the two-layer network using SGD on toy data, train the two-layer network using SGD on CIFAR-10 data with learning rate decay, visualize weights of the network, tune hyperparameters using the validation set, visualize the weights of the best network, evaluate your final trained network on the test set.`

- Q5: [Higher Level Representations: Image Features](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/2019/html/assignment1/features.html)

    `load CIFAR-10 data & prepare train val and test sets, feature extraction: compute a Histogram of Oriented Gradients and a color histogram using the hue channel in HSV color space, train and evaluate SVM on extracted features, visualize misclassified images, train a two-layer neural network on image features, tune hyperparameters, evaluate your final trained network on the test set.`

### **Assignment 2:**

- Q1: [Fully-connected Neural Network]()

- Q2: [Batch Normalization ]()

- Q3: [Dropout ]()

- Q4: [Convolutional Networks]()

- Q5: [PyTorch / TensorFlow on CIFAR-10]()

### **Assignment 3:**

- Q1: [Image Captioning with Vanilla RNNs]()

- Q2: [Image Captioning with LSTMs]()

- Q3: [Network Visualization: Saliency maps, Class Visualization, and Fooling Images]()

- Q4: [Style Transfer ]()

- Q5: [Generative Adversarial Networks]()















