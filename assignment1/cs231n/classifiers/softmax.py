import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    logits = np.matmul(X,W) #theta NxC
    logitGrad = np.zeros_like(logits) #NxC
    normalizedLogits = logits - np.amax(logits, axis = 1).reshape(-1,1) #theta NxC
    expLogits = np.exp(normalizedLogits) #NxC
    sumExp = np.sum(expLogits, axis=1).reshape(-1,1)  #Nx1
    numTrain = X.shape[0]
    numClasses = W.shape[1]
    for i in range(numTrain):
        for j in range(numClasses):
            predProb = expLogits[i, j]/sumExp[i,0]
            if y[i] == j:
                loss -= np.log(predProb)
                logitGrad[i,j] = predProb - 1
            else:
                logitGrad[i,j] = predProb
    loss = loss/numTrain + reg*np.sum(W*W)
            
    dW = np.matmul(X.T, logitGrad)/numTrain + 2*reg*(W)
            
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    numTrain = X.shape[0]
    logits = np.matmul(X,W) #theta NxC
    logitGrad = np.zeros_like(logits) #NxC
    shift = np.amax(logits, axis = 1).reshape(-1,1)
    normalizedLogits = logits - shift #theta NxC
    expLogits = np.exp(normalizedLogits) #NxC
    sumExp = np.sum(expLogits, axis=1).reshape(-1,1)  #Nx1
    probs = expLogits/sumExp #probabilities; classwise of each sample
    correctClass = probs[np.arange(numTrain), y]
    loss = -np.sum(np.log(correctClass))/numTrain + reg*np.sum(W*W)
    logitGrad = probs
    logitGrad[np.arange(numTrain), y] -= 1
    dW = np.matmul(X.T, logitGrad)/numTrain + 2*reg*W
    
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW