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
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    score = X[i].dot(W)
    score -= score.max()
    score_exp_sum = np.sum(np.exp(score))
    CE = np.exp(score[y[i]])
    loss += -np.log(CE / score_exp_sum)

    dW[:, y[i]] += -(score_exp_sum -CE) / (score_exp_sum) * X[i]
    for j in range(num_class):
      if j == y[i]:
        continue
      dW[:, j] += np.exp(score[j]) / score_exp_sum *X[i]
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  score = X.dot(W)
  score -= score.max()
  score_exp_sum = np.sum(np.exp(score), axis=1, keepdims=True)
  CE = np.exp(score) / score_exp_sum

  loss = -np.sum(np.log(CE[np.arange(num_train), y]))/num_train + reg * np.sum(W * W)

  dS = CE
  dS[np.arange(num_train), y] -= 1
  dW = X.T.dot(dS) / num_train + 2 * reg * W

  return loss, dW

