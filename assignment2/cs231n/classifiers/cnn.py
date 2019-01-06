import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               use_batchnorm=False, dtype=np.float32):
    """
    Initialize a new network.
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.use_bn = use_batchnorm
    self.reg = reg
    self.dtype = dtype
    self.stride = 1
    self.pad = (filter_size - 1) / 2
    self.h_pool = 2
    high0 = 1 + (input_dim[1] + 2 * self.pad - filter_size) / self.stride
    wide0 = 1 + (input_dim[2] + 2 * self.pad - filter_size) / self.stride
    high = 1 + (high0 - self.h_pool) / self.h_pool
    wide = 1 + (wide0 - self.h_pool) / self.h_pool
    hid_input_dim = num_filters * high * wide

    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
    self.params['W1'] /= np.sqrt(input_dim[0] * filter_size * filter_size / 2)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(hid_input_dim, hidden_dim)
    self.params['W2'] /= np.sqrt(hid_input_dim / 2)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes)
    self.params['W3'] /= np.sqrt(hidden_dim / 2)
    self.params['b3'] = np.zeros(num_classes)

    if self.use_bn:
      self.params['gamma1'] = np.ones(num_filters*high0*wide0)
      self.params['beta1'] = np.zeros(num_filters*high0*wide0)
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    conv_param = {'stride': self.stride, 'pad': self.pad}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    if self.use_bn:
      mode = 'test' if y is None else 'train'
      bn_param = [{'mode': mode}, {'mode': mode}]
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      gamma2, beta2 = self.params['gamma2'], self.params['beta2']
      H1, cache1 = conv_bn_relu_pool_forward(X, W1, b1, conv_param, gamma1, beta1, bn_param[0], pool_param)
      H2, cache2 = aff_bn_relu_forward(H1, W2, b2,  gamma2, beta2, bn_param[1])
    else:
      H1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      H2, cache2 = affine_relu_forward(H1, W2, b2)
    scores, cache3 = affine_forward(H2, W3, b3)

    if y is None:
        return scores

    loss, grads = 0.0, {}
    loss, dH3 = softmax_loss(scores, y)
    loss += self.reg * 0.5 * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    dH2, dW3, db3 = affine_backward(dH3, cache3)
    if self.use_bn:
      dgamma2, dbeta2, dH1, dW2, db2 = aff_bn_relu_backward(dH2, cache2)
      dgamma1, dbeta1, ___, dW1, db1 = conv_bn_relu_pool_backward(dH1, cache1)
      grads = {'gamma1': dgamma1, 'beta1': dbeta1, 'gamma2': dgamma2, 'beta2': dbeta2}
    else:
      dH1, dW2, db2 = affine_relu_backward(dH2, cache2)
      ___, dW1, db1 = conv_relu_pool_backward(dH1, cache1)
    dW3 += self.reg * W3
    dW2 += self.reg * W2
    dW1 += self.reg * W1
    grads['W1'], grads['W2'], grads['W3'] = dW1, dW2, dW3
    grads['b1'], grads['b2'], grads['b3'] = db1, db2, db3

    return loss, grads
