import numpy as np
from random import shuffle

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
    D, C = W.shape
    N = y.shape[0]
    
    scores = np.zeros((N, C))
    for sample_idx in range(N):
        for class_idx in range(C):
            for coord_idx in range(D):
                scores[sample_idx, class_idx] += X[sample_idx, coord_idx] * W[coord_idx, class_idx]
                
    probs = np.zeros_like(scores)
    for sample_idx in range(N):
        softmax_den = 0
        for class_idx in range(C):
            exponent = np.exp( scores[sample_idx, class_idx] )
            probs[sample_idx, class_idx] = exponent
            softmax_den += exponent
        for class_idx in range(C):
            probs[sample_idx, class_idx] /= softmax_den
            
    # Compute loss
    log_probs = 0
    for sample_idx in range(N):
        true_class_idx = y[sample_idx]
        log_probs += np.log( probs[sample_idx, true_class_idx] )
    
    data_loss = -(1 / N) * log_probs
    
    reg_loss = 0
    if reg != 0:
        for coord_idx in range(D):
            for class_idx in range(C):
                reg_loss += W[coord_idx, class_idx] ** 2
            
    reg_loss = 0.5 * reg * reg_loss
        
    loss = data_loss + reg_loss

    #Compute grad
    dscores = probs
    
    for sample_idx in range(N):
        true_class_idx = y[sample_idx]
        dscores[sample_idx, true_class_idx] -= 1
        
    for sample_idx in range(N):
        for class_idx in range(C):
            dscores[sample_idx, class_idx] /= N
                
    for coord_idx in range(D):
        for class_idx in range(C):
            for sample_idx in range(N):          
                dW[coord_idx, class_idx] += X.T[coord_idx, sample_idx] * dscores[sample_idx, class_idx]
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
    N = y.shape[0]
    
    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(N),y])
    data_loss = np.sum(correct_logprobs)/N
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    
    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg*W 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
