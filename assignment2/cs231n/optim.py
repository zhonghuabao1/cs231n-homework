#coding=utf-8
import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""
def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  next_w = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  v = config['momentum'] * v - config['learning_rate'] * dw
  next_w = w + v

  config['velocity'] = v

  return next_w, config

def Nesterov(w, dw, config=None):
    
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  next_w = None
  mu = config['momentum']
  v_pred = v
  v = mu * v - config['learning_rate'] * dw
  next_w = w + v + mu * (v - v_pred)

  config['velocity'] = v

  return next_w, config


def rmsprop(x, dx, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))

  next_x = None
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of x   #
  # in the next_x variable. Don't forget to update cache value stored in      #  
  # config['cache'].                                                          #
  #############################################################################
  decay_rate = config['decay_rate']
  cache = config['cache']
  lr = config['learning_rate']
  eps = config['epsilon']
  cache = decay_rate * cache + (1 - decay_rate) * dx**2
  next_x = x - lr * dx / (np.sqrt(cache) + eps)
   
  config['cache'] = cache

  return next_x, config


def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                       
  #############################################################################
  beta1 = config['beta1']
  beta2 = config['beta2']
  m = config['m']
  v = config['v']
  t = config['t']
  lr = config['learning_rate']
  eps = config['epsilon']
  m = beta1 * m + (1 - beta1) * dx
  v = beta2 * v + (1 - beta2) * dx**2
  t = t + 1
  m_bias = m / (1 - beta1**t)
  v_bias = v / (1 - beta2**t)
  next_x = x - lr * m_bias / (np.sqrt(v_bias) + eps)
   
  config['m'] = m
  config['v'] = v
  config['t'] = t
  
  return next_x, config

def middle(x, dx, p, config=None):
# 根据最近三次的梯度方向判断，有3种情况，遇见低谷就采用2分法
# 遇见斜坡则用 sgd
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('d1', np.zeros_like(x))
  config.setdefault('d2', np.zeros_like(x))
  config.setdefault('dx1', dx)
  config.setdefault('dx2', dx)
  lr = config['learning_rate']
  d1 = config['d1']
  d2 = config['d2']
  dx1 = config['dx1']
  dx2 = config['dx2']
    
  dx3 = dx > 0
  judge23 = dx2 == dx3
  judge12 = dx2 == dx1
  d3 = -d2/2 * (judge23==0)   # 第一次遇见波谷
  d3 += lr * dx * judge23 * judge12  # 斜坡用 sgd
  d3 += -(d1 + d2)/2 * judge23 * (judge12==0) # 在波谷震动
  next_x = x - d3

  d1 = d2
  d2 = d3
  dx1 = dx2
  dx2 = dx3
  config['d1'] = d1
  config['d2'] = d2
  config['dx1'] = dx1
  config['dx2'] = dx2

  return next_x, config