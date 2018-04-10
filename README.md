# cs231n
Visual recognition with deep learning

assignment1:

knn.ipynb - kNN classifier on CIFAR-10 in numpy

svm.ipynb - SVM classifier on CIFAR-10 in numpy

          fully vectorized loff function for SVM
          
          fully vectorized implementation of its analytic gradient
          
          implementing sanity checks using numerical gradient
          
          optimization of the entire network using SGD
          
          visualizing the final learned weights
          
softmax.ipynb - softmax classifier on CIFAR-10 in  numpy

          fully vectorized loss function for softmax
          
          fully vectorized implementation of its analytic gradient
          
          implementing sanity checks using numerical gradient
          
          optimization of the entire classifier using SGD
          
          visualizing the final learned weights
          
two_layer_net.ipynb - Two layered neural network classifier on CIFAR-10 in numpy

features.ipynb - Two ayered neural network classifier on CIFAR-10 in numpy
          
          Uses concatenation of Histogram of oriented gradients and color histogram as input features
          
          This is to show that feature representation of same data affects the classifier performance
          
          If the network was deep these features are learned by the network
          
          

assignment2:

FulllyConnectedNets.ipynb
          
          Implementing fully-connected networks using a more modular approach. For each layer we will implement
          a forward and a backward function. The forward function will receive inputs, weights, and other 
          parameters and will return both an output and a cache object storing data needed for the backward pass
          This is a comparison suite for various update rules and activation functions as well
          
BatchNormalization.ipynb

          https://arxiv.org/pdf/1502.03167.pdf implemented in layers of a network with fully connected layers.
          Numpy implementation
          
Dropout.ipynb

          https://arxiv.org/pdf/1207.0580.pdf implemented in layers of a network with fully connected layers.
          Numpy implementation
          
ConvolutionalNetworks.ipynb

          implement several layer types that are used in convolutional networks then use these layers to train
          a convolutional network on the CIFAR-10 dataset. Numpy from scratch implementation of CNNs
          
          Viualize the filters that CNN learns on CIFAR-10
        
TensorFlow.ipynb
          
          Learning the basics of Tensorflow through this notebook and then use Tensorflow to implement a multulayered CNN
          to train classifier on CIFAR-10
          

assignment3:

RNN_Captioning.ipynb

          Image captioning with vanilla RNN. Implement a vanilla recurrent neural networks and use them it to train a model
          that can generate novel captions for images
          
LSTM_Captioning.ipynb
      
          Image captioning with LSTM. Implement a LSTM recurrent neural networks and use them it to train a model
          that can generate novel captions for images
          

NetworkVisualization-TensorFlow.ipynb

          Image generation explored through three different techniques. Tensorflow
            1. Saliency Maps
            2. Fooling Images
            3. Class Visualization
            
            
 StyleTransfer-Tensorflow.ipynb
 
          take two images, and produce a new image that reflects the content of one but the artistic "style"
          of the other. This is done by first formulating a loss function that matches the content and
          style of each respective image in the feature space of a deep network, and then performing gradient
          descent on the pixels of the image itself.
          
          This is also an introduction to gram matrices - A computationally cheap representation of correlation
          matrix. Captures the recurring patterns in an image
          
GANs-TensorFlow.ipynb

        Implementaion of https://arxiv.org/abs/1406.2661 in tensorflow to generate images that are similar to the training
        dataset. This is done by creating a generative network and an adversary that calls its generated images as bogus.
        Over time both improve against each other and hence as standalone networks
          
  
          
          




