# Convolutional Neural Networks for Visual Recognition

This is an overview of implementations for [**CS231n : Convolutional Neural Networks for Visual Recognition**](http://cs231n.stanford.edu/2019/) by Stanford University.

## Resources

&nbsp;&nbsp;&nbsp;&nbsp;Stanford CS231n Lectures [Playlist.](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

&nbsp;&nbsp;&nbsp;&nbsp;Source code as a zip file:

- [Assignment1](cs231n.github.io/assignments/2019/spring1819_assignment1.zip)
- [Assignment2](cs231n.github.io/assignments/2019/spring1819_assignment2.zip)
- [Assignment3](cs231n.github.io/assignments/2019/spring1819_assignment3.zip)

&nbsp;&nbsp;&nbsp;&nbsp;Environment Setup [Instructions.](https://cs231n.github.io/setup-instructions/)

## Implementations (Author: Rohit Jain)

`python` `numpy` `matplotlib` `jupyter` `PyTorch`

>
> All the implementations were done with the following setup:
> - *Processor* - Intel Core 2 Duo CPU P8600 @2.40 GHz
> - *RAM* - 4 GB
> - *OS* - Windows 7
>
> HTML output of jupyter notebooks for my implementattion of the year 2019 version of CS231n is avaibable [here](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/tree/main/2019/html_new) for quick reference. Assignments were re-run on a different setup and various execution time in output are from that re-run on different configuration. 
>
> You can also find the output for assignment1 from the initial setup sitting [here](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/tree/main/2019/html)
>
> Code for my implementations is not posted on this public repository.

### **Assignment 1:**

- Q1: [k-Nearest Neighbor classifier](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment1/knn.html)

    `load and visualize CIFAR-10 dataset, subsample test data & train data, compute visualize & vectorize distance matrix, implement k-Nearest Neighbor classifier, Perform k-fold cross validation to find the best value of k, retrain and retest the classifier for the best value of k.`

- Q2: [Training a Support Vector Machine](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment1/svm.html)

    `CIFAR-10 data loading and preprocessing, prepare train val and test sets, implement a fully-vectorized loss function for the SVM, implement the fully-vectorized expression for its analytic gradient, check your implementation using numerical gradient, use a validation set to tune the learning rate and regularization strength, optimize the loss function with SGD, visualize the final learned weights.`


- Q3: [Implement a Softmax classifier](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment1/softmax.html)

    `CIFAR-10 data loading and preprocessing, prepare train val dev and test sets, implement a fully-vectorized loss function for the Softmax classifier, implement the fully-vectorized expression for its analytic gradient, check your implementation with numerical gradient, use a validation set to tune the learning rate and regularization strength, optimize the loss function with SGD, visualize the final learned weights.`

- Q4: [Two-Layer Neural Network](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment1/two_layer_net.html)

    `implement forward pass to compute scores & data and regularization loss, implement backward pass to compute gradient of loss and perform numeric gradient checking, train the two-layer network using SGD on toy data, train the two-layer network using SGD on CIFAR-10 data with learning rate decay, visualize weights of the network, tune hyperparameters using the validation set, visualize the weights of the best network, evaluate your final trained network on the test set.`

- Q5: [Higher Level Representations: Image Features](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment1/features.html)

    `load CIFAR-10 data & prepare train val and test sets, feature extraction: compute a Histogram of Oriented Gradients and a color histogram using the hue channel in HSV color space, train and evaluate SVM on extracted features, visualize misclassified images, train a two-layer neural network on image features, tune hyperparameters, evaluate your final trained network on the test set.`

### **Assignment 2:**
>
> For the last part - Q5, you can work in either TensorFlow or PyTorch. 
> 
> PyTorch implementation requires the following as specified in requirements.txt for assignment2
>
> - torch==1.0.1.post2 
> - torchvision==0.2.2.post3
>
>PyPI distribution for these versions are not available for windows, you can download the compatible distribution from [here](https://download.pytorch.org/whl/torch_stable.html) and do a pip install.
> 
> I had dowloaded and installed [these](https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/tree/main/2019/wheel) `Python3.7 Windows` distributions from the link specified above.
>

- Q1: [Fully-connected Neural Network](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment2/FullyConnectedNets.html)

    `load preprocessed CIFAR-10 data, implement affine_forward and affine_backward function, implement forward pass and backward pass for ReLU activation function, test implementations using numeric gradient checking, verify affine_relu_forward affine_relu_backward and loss function, implement TwoLayerNet, use a Solver instance to train a TwoLayerNet, visualize training loss and train / val accuracy, implement a fully-connected network with an arbitrary number of hidden layers and perform sanity checks, implement SGD+momentum &  RMSProp & Adam(with the bias correction mechanism) - update rules, train a deep network with these new update rules, visualize training loss and train / val accuracy with different update rules.`

- Q2: [Batch Normalization ](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment2/BatchNormalization.html)

    `load preprocessed CIFAR-10 data, implement batch normalization forward pass, Check the training-time forward pass before and after batch normalization, implement backward pass for batch normalization and perform gradient check, implement alternative simplified batch normalization backward pass, compare both of the batch norm implementations, train a six-layer network on a subset of 1000 training examples both with and without batch normalization and visualize the results from trained networks, train 8-layer networks both with and without batch normalization using different scales for weight initialization, plot results of weight scales, train 6-layer networks both with and without batch normalization using different batch sizes, plot results of batch sizes, implement forward pass and backward pass for layer normalization and Check training-time of forward pass before and after layer normalization, run the batch size experiment on layer normalization.` 

- Q3: [Dropout ](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment2/Dropout.html)

    `load preprocessed CIFAR-10 data, implement forward pass of dropout for both training and testing mode, implement backward pass for dropout and perform numerical gardient check, mofify the FullyConnectedNet implementation to use dropout and numerically gradient check the implementation, train a pair of two-layer networks on 500 training examples -  one will use no dropout and one will use a keep probability of 0.25, visualize the training and validation accuracies of the two networks over time.`

- Q4: [Convolutional Networks](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment2/ConvolutionalNetworks.html)

    `load preprocessed CIFAR-10 data, implement forward pass for convolution layer, perform image processing-'grayscale conversion and edge detection' via convolutions, visualize the results as a sanity check, implement backward pass for convolution operation and perform numerical gradient check, Implement forward pass and backward pass for max-pooling operation and perform numerical gradient check, compare the performance of naive and fast-'depends on a Cython extension' versions of convolution and max-pooling layers, implement the ThreeLayerConvNet and sanity check loss and perform numerical gradient check, train your model to overfit a small dataset and visualize training loss and train / val accuracy, train the three-layer convolutional network for one epoch, visualize first-layer convolutional filters from the trained network, implement forward pass and backward pass for spatial batch normalization and perform numerical gradient check,implement forward pass for group normalization and backward pass for spatial group normalization and perform numerical gradient check.`

- Q5: [PyTorch / TensorFlow on CIFAR-10](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment2/PyTorch.html)

    `iterate CIFAR-10 dataset through dataloader and form train and val sets, test the flatten function which reshapes image data, implement the forward pass of a three-layer convolutional network with defined architecture in Barebones PyTorch, initialize the weight matrices models, Check the accuracy of a classification model, train the model on CIFAR dataset using stochastic gradient descent without momentum, use torch.functional.cross_entropy to compute the loss, train a Two-Layer Network and check accuracy, train a three-layer convolutional network on CIFAR, set up the three-layer ConvNet and implement forward function with the defined architecture in PyTorch Module API, check the classification accuracy of neural network, train a model on CIFAR-10 using the PyTorch Module API, train a Two-Layer Network in PyTorch Module API, train a Three-Layer ConvNet with PyTorch Module API, rewrite and train two-layer fully connected network example with PyTorch Sequential API, rewrite the 2-layer ConvNet with bias with PyTorch Sequential API, experiment with architectures, hyperparameters, loss functions, and optimizers to train a model that achieves at least 70% accuracy on the CIFAR-10 validation set within 10 epochs.`

    >Assignment2 also have a PyTorch [tutorial](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment2/pytorch_tutorial.html)


    

### **Assignment 3:**

- Q1: [Image Captioning with Vanilla RNNs](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment3/RNN_Captioning.html)

    `load Microsoft COCO dataset which is stored in HDF5 format, sample a minibatch and show the images and captions, implement forward pass for a single timestep of a vanilla RNN, implement vanilla RNN step backward, perform neumerical gradient check, implement a RNN that processes an entire sequence of data, implement backward pass for vanilla RNN which runs back-propagation over the entire sequence, implement word_embedding forward to convert words into vectors. Implement backward pass for word embedding function, perform sanity check for temporal softmax loss, build an image captioning model and perform gradient check, overfit a small sample of 100 training examples, implement test-time sampling, sample from your overfitted model on both training and validation data.`

- Q2: [Image Captioning with LSTMs](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment3/LSTM_Captioning.html)

   `Load MS-COCO data from disk, Implement forward pass for a single timestep of an LSTM, Implement backward pass for a single LSTM timestep, perform numerical gradient check, implement forward function to run an LSTM forward on an entire timeseries of data, Implement the backward pass for an LSTM over an entire timeseries of data, update the implementation of the loss method of the Captioning RNN, overfit an LSTM captioning model on a small dataset, modify sampling method  to handle the case where self.cell_type is lstm, sample from your overfit LSTM model on some training and validation set samples.`

- Q3: [Network Visualization: Saliency maps, Class Visualization, and Fooling Images](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment3/NetworkVisualization-PyTorch.html)

  `download and load the pretrained SqueezeNet model, visualize some of the images along with their ground-truth labels from the ImageNet ILSVRC 2012 Classification dataset, compute a class saliency map using the model for given images and labels, visualize some class saliency maps on example images from ImageNet validation set,  implement gradient ascent over the image to generate a fooling image that the model classifies as target class, generate a fooling image, visualize the original image the fooling image and the difference between them, starting with random noise implement gradient ascent on a target class to generate an image that the network will recognize as the target class, visualize images generated for class visualization, try out class visualization on other classes.`

- Q4: [Style Transfer ](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment3/StyleTransfer-PyTorch.html)

  `load the pre-trained SqueezeNet model, compute the content loss for style transfer, test the content loss, compute the Gram matrix from feature, test the Gram matrix code, compute the style loss at a set of layers, test the style loss implementation, Compute total variation loss, test the TV loss implementation, run style transfer, try out and visualize style_transfer on the three different parameter sets, try out feature inversion to reconstruct an image from its feature representation, try out texture synthesis from scratch.`

- Q5: [Generative Adversarial Networks](http://htmlpreview.github.io/?https://github.com/r-jain/Convolutional-Neural-Networks-for-Visual-Recognition/blob/main/2019/html_new/assignment3/Generative_Adversarial_Networks_PyTorch.html)

  `prepare train / val sets of mnist dataset through PyTorch MNIST wrapper, process image data, build a PyTorch model for Discriminator implementing the specified architecture, verify the number of parameters in the discriminator, build a PyTorch model for Generator implementing the specified architecture,  verify the number of parameters in the Generator, implement binary cross-entropy loss function, implement loss function for the Discriminator, implement loss function for the Generator, test generator and discriminator loss, construct an Adam optimizer for the model with the desired hyperparameters, run a GAN, compute the Least-Squares GAN loss for the Discriminator, computes the Least-Squares GAN loss for the Generator, check loss functions, run Least Squares GAN, build a PyTorch model for the Deeply Convolutional GAN Discriminator implementing specified architecture, check the number of parameters in the classifier, Build a PyTorch model implementing the DCGAN Generator implementing specified architecture, check the number of parameters in the Generator as a sanity check, run DCGAN.`















