# -*- coding:utf-8 -*-
from cs231n.layers import *
# from cs231n.fast_layers import *

################ affine #####################################################

def affine_relu_forward(x, w, b):
  """"""
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  
  return out, cache


def affine_relu_backward(dout, cache):
  """"""
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache) 
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def aff_bn_forward(x, w, b, gamma, beta, bn_param):
  """"""
  out, aff_cache = affine_forward(x, w, b)
  out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param) # batch normal 部分
  cache = (aff_cache, bn_cache)
  return out, cache


def aff_bn_backward(dout, cache):
  """"""
  (aff_cache, bn_cache) = cache
  dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)  # batch normal梯度传递
  dx, dw, db = affine_backward(dout, aff_cache)  # 全连接 梯度传递
  return dgamma, dbeta, dx, dw, db


def aff_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """"""
  out, aff_cache = affine_forward(x, w, b)
  out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param) # batch normal 部分
  out, relu_cache = relu_forward(out) 
  cache = (aff_cache, bn_cache, relu_cache)
  return out, cache


def aff_bn_relu_backward(dout, cache):
  """"""
  (aff_cache, bn_cache, relu_cache) = cache
  dout = relu_backward(dout, relu_cache)
  dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)  # batch normal梯度传递
  dx, dw, db = affine_backward(dout, aff_cache)  # 全连接 梯度传递
  return dgamma, dbeta, dx, dw, db


################ convolution #####################################################

def conv_relu_forward(x, w, b, conv_param):
  """"""
  a, conv_cache = conv_forward_vector(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """"""
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_vector(da, conv_cache)
  return dx, dw, db

def conv_pool_forward(x, w, b, conv_param, pool_param):
  """ """
  a, conv_cache = conv_forward_vector(x, w, b, conv_param)
  out, pool_cache = max_pool_forward_vector(a, pool_param)
  cache = (conv_cache, pool_cache)
  return out, cache


def conv_pool_backward(dout, cache):
  """  """
  conv_cache, pool_cache = cache
  ds = max_pool_backward_vector(dout, pool_cache)
  dx, dw, db = conv_backward_vector(ds, conv_cache)
  return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """ """
  a, conv_cache = conv_forward_vector(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_vector(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_vector(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_vector(da, conv_cache)
  return dx, dw, db


def conv_bn_relu_pool_forward(x, w, b, conv_param, gamma, beta, bn_param, pool_param):
  """"""
  a, conv_cache = conv_forward_vector(x, w, b, conv_param)
  s, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(s)
  out, pool_cache = max_pool_forward_vector(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache


def conv_bn_relu_pool_backward(dout, cache):
  """"""
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_vector(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_vector(da, conv_cache)
  return dgamma, dbeta, dx, dw, db
