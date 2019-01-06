# -*- coding: utf-8 -*-
import numpy as np


def affine_forward(x, w, b):
  """全连接"""
  X = np.reshape(x, (x.shape[0], -1))
  out = X.dot(w) + b
  cache = (x, w)

  return out, cache


def affine_backward(dout, cache):
  """全连接"""
  x, w = cache
  dx = np.dot(dout, w.T)
  dx = dx.reshape(x.shape)
  x_row = x.reshape(x.shape[0], -1)
  dw = np.dot(x_row.T, dout)
  db = np.sum(dout, 0)

  return dx, dw, db


def relu_forward(x, leaky=0):
  """Leaky relu"""
  out = x.copy()
  neg_x = x <= 0
  out[neg_x] *= leaky
  cache = neg_x, leaky
  return out, cache


def relu_backward(dout, cache):
  """Leaky relu"""
  neg_x, leaky = cache
  dx = dout.copy()
  dx[neg_x] *= leaky
  return dx


def conv_forward_vector(x, w, b, conv_param):
  """为加快计算，x.shape 统一是 [C,N,H,W]"""
  if x.shape[1] == 3:  # 第一层输入图像的时候检查
    x = x.transpose(1, 0, 2, 3)
  C, N,  H,  W = x.shape
  F, _, HH, WW = w.shape
  stride,  pad = conv_param['stride'], conv_param['pad']
  # Check dimensions
  H += 2 * pad
  W += 2 * pad
  assert (H - HH) % stride == 0, 'width does not work'
  assert (W - WW) % stride == 0, 'height does not work'
  out_h = 1 + (H - HH) / stride
  out_w = 1 + (W - WW) / stride
  x_p_shape = (C, N, H, W)
  w_shape = (F, C, HH, WW)
  x_pad = np.zeros(x_p_shape)
  if pad > 0:
    x_pad[:, :, pad:-pad, pad:-pad] = x
  else:
    x_pad = x
  x_new = np.zeros([C, HH, WW, N, out_h, out_w])  # 将x变换后的形状，预分配内存
  for i in xrange(HH):
    for j in xrange(WW):  # 例如，i=0，j=0时，把滤板左上角的元素集合成一个矩阵
      x_new[:,i,j] = x_pad[:,:,i::stride,j::stride][:,:,:out_h,:out_w]
  x_r = x_new.reshape(C * HH * WW, -1)
  w_r = w.reshape(F, -1)
  out_new = np.dot(w_r, x_r) + b[:, np.newaxis]
  out = out_new.reshape(F, N, out_h, out_w)
  # out = out_new.reshape(F, N, out_h, out_w).transpose(1,0,2,3)
  cache = (x_r, x_p_shape, w_r, w_shape, conv_param)

  return out, cache


def conv_backward_vector(dout, cache):
  """和 forward 对应"""
  x_r, x_p_shape, w_r, w_shape, conv_param = cache
  stride, pad = conv_param['stride'], conv_param['pad']
  F, C, HH, WW = w_shape
  _, N, out_h, out_w = dout.shape

  dx_pad = np.zeros(x_p_shape)
  # dout_r = dout.transpose(1,0,2,3).reshape(F, -1); N = dout.shape[0]
  dout_r = dout.reshape(F, -1)
  db = np.sum(dout_r, axis=1)
  dw = np.dot(dout_r, x_r.T).reshape(F, C, HH, WW)
  dx_r = np.dot(w_r.T, dout_r).reshape(C, HH, WW, N, out_h, out_w)
  for i in xrange(HH):
    for j in xrange(WW):
      dx_pad[:,:,i::stride,j::stride][:,:,:out_h,:out_w] += dx_r[:,i,j]
  # if pad > 0:
  #   dx = dx_pad[:, :, pad:-pad, pad:-pad]
  #   # dx = dx_pad[:, :, pad:-pad, pad:-pad].transpose(1,0,2,3)
  # else:
  #   dx = dx_pad
  dx = dx_pad[:, :, pad:-pad, pad:-pad] if pad else dx_pad

  return dx, dw, db


def max_pool_forward_vector(x, pool_param):
  """ """
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  if HH < 2:
    return x, (None, None, pool_param)
  N, C, H, W = x.shape
  assert (H - HH) % stride == 0, 'Invalid height'
  assert (W - WW) % stride == 0, 'Invalid width'
  same_size = HH == WW == stride
  tiles = H % HH == 0 and W % WW == 0
  condition = same_size and tiles
  if condition:  # 第一种情况
    x_new = x.reshape(N, C, H/HH, HH, W/WW, WW)
    out = x_new.max(axis=3).max(axis=4)
  else:                   # 第二种情况
    out_h = 1 + (H - HH) / stride
    out_w = 1 + (W - WW) / stride
    x_new = np.zeros([N, C, out_h, HH, out_w, WW])  # 将x变换后的形状，预分配内存
    for i in xrange(HH):
      for j in xrange(WW):
        x_select = x[:, :, i::stride, j::stride]  # 例如，i=0，j=0时，把滤板左上角的元素集合成一个矩阵
        x_new[:,:,:,i,:,j] = x_select[:, :, :out_h, :out_w]
    out = x_new.max(axis=3).max(axis=4)  # 最大池化
  cache = (x_new, out, stride, condition)

  return out, cache

 
def max_pool_backward_vector(dout, cache):
  """"""
  x_new, out, stride, condition = cache
  N, C, out_h, HH, out_w, WW = x_new.shape
  if HH < 2:
    return dout
  dx_new = np.zeros_like(x_new)
  for i in xrange(HH):
    for j in xrange(WW):
      x_select = x_new[:,:,:,i,:,j]
      dx_new[:, :, :, i, :, j] = dout * (x_select == out)

  if condition:
    dx = dx_new.reshape(N, C, out_h*HH, out_w*WW)
  else:
    H = stride * (out_h - 1) + HH
    W = stride * (out_w - 1) + WW
    dx = dx_new.reshape(N, C, H, W)

  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.
    At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:
       running_mean = momentum * running_mean + (1 - momentum) * sample_mean
       running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.
  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  N, D = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-10)
  momentum = bn_param.get('momentum', 0.9)
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
  out, cache = None, None
  if mode == 'train':
    E = np.mean(x, 0)
    bn_param['running_mean'] = momentum * running_mean + (1-momentum) * E
    x -= E                   # 均值变为 0
    Var = np.var(x, 0)
    bn_param['running_var'] = momentum * running_var + (1-momentum) * Var
    Var = 1 / np.sqrt(Var + eps)
    x *= Var                       # 方差变为 1
    # print x.shape, gamma.shape
    out = gamma * x + beta            # Covariate Shift
    cache = (E, Var, gamma, x)
  elif mode == 'test':                                                      #
    out = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * out + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  N = dout.shape[0]
  E, Var, gamma, x = cache
  dgamma = np.sum(dout * x, 0)
  dbeta = np.sum(dout, 0)
  dx_ = dout * gamma           # H * gamma + beta = H~
  dVar = -np.sum(dx_ * E, 0) * (Var**3) / 2   # 在批量方向加起来
  dE1 = dx_ * Var
  dE2 = dVar * 2 * E / N
  dE = np.mean(dE1, 0) + np.mean(dE2, 0)
  dx = dE1 + dE2 - dE

  return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass               """
  # TODO: Implement the forward pass for spatial batch normalization.         #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  N, C, H, W = x.shape
  x_new = x.transpose(0, 2, 3, 1).reshape(-1, C)
  # x_new = x.reshape(N, -1)另一种batchnorm
  out_new, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
  out = out_new.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  # out = out_new.reshape(N, C, H, W)另一种batchnorm

  return out, cache

def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  # TODO: Implement the backward pass for spatial batch normalization.        #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  N, C, H, W = dout.shape
  dout_new = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  # dout_new = dout.reshape(N, -1)另一种batchnorm
  dx_new, dgamma, dbeta = batchnorm_backward(dout_new, cache)
  dx = dx_new.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  # dx = dx_new.reshape(N, C, H, W)另一种batchnorm

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask, out = None, None
  if mode == 'train':
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    mask = (np.random.randn(*x.shape) < p) / p
    out = x * mask
  elif mode == 'test':
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dx = None
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)  # np.newaxis 插入新维度，方便广播
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N

  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  x = np.reshape(x, (N, -1))
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True) 
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N             # shape(dx): （N， classes）

  return loss, dx

#####################################################################################
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  high = 1 + (H + 2 * pad - HH) / stride
  wide = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros([N, F, high, wide])
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')  # 边缘扩展0
  for i in xrange(high):  # 下一层的行
    index_h = i * stride  # 上一层的行，起点
    for j in xrange(wide):  # 下一层的列
      index_w = j * stride  # 上一层的列，起点
      for k in xrange(F):  # 滤波器个数
        small_x = x_pad[:, :, index_h:index_h + HH, index_w:index_w + WW]
        out[:, k, i, j] = np.sum(small_x * w[k], axis=(1, 2, 3)) + b[k]

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w  high = 1 + (H + 2 * pad - HH) / stride
  wide = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros([N, F, high, wide])
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  _, _, high, wide = dout.shape

  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')  # 边缘扩展0
  dx, dw, db, dx_pad = None, np.zeros_like(w), np.zeros(F), np.zeros_like(x_pad)
  for i in xrange(high):  # 下一层的行
    index_h = i * stride  # 上一层的行，起点
    for j in xrange(wide):  # 下一层的列
      index_w = j * stride  # 上一层的列，起点
      for k in xrange(F):  # 滤波器个数
        db[k] += np.sum(dout[:, k, i, j])
        x_pad_index = x_pad[:, :, index_h:index_h + HH, index_w:index_w + WW]
        x_pad_index = dout[:, k, i, j] * x_pad_index.transpose(1, 2, 3, 0)  # 在‘数量N’维度相容， 广播
        dw[k] += np.sum(x_pad_index, axis=3)  # 数量N 相加
        w_broadcast = dout[:, k, i, j] * w[k, :, :, :, np.newaxis]
        dx_pad[:, :, index_h:index_h + HH, index_w:index_w + WW] += w_broadcast.transpose(3, 0, 1, 2)

  dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
  return dx, dw, db
