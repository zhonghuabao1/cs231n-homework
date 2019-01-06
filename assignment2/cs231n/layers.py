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


def conv_forward(x, w, b, conv_param):
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


def conv_backward(dout, cache):
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


def bl_pool_forward(x1, x2, conv_param):
  """"""
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


def bl_pool_backward(dout, cache):
  """"""
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


def max_pool_forward(x, pool_param):
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


def max_pool_backward(dout, cache):
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

