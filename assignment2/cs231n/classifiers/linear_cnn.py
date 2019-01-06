# -*- coding: utf-8 -*-
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from scipy import io


class LinearCNN(object):
    def __init__(self, input_dim=(3, 32, 32),
                 conv_sets=[(32,3,3,2), (32,3,3,2)],
                 pool_params=[(2,2,2), (2,2,2)],
                 aff_dim=[100, 100], num_classes=10,
                 reg=0.0, use_batchnorm=True,
                 reset=False, dtype=np.float32, p=1/2):
        self.p = p
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.params, self.conv_params, self.pool_params = {}, {}, {}
        self.conv_layers = len(conv_sets)
        self.aff_layers = len(aff_dim) + 1
        self.num_layers = self.conv_layers + self.aff_layers
        # conv_layers initial
        C, H, W = input_dim
        for i in xrange(self.conv_layers):  # 卷积核初始化
            j = i + 1
            F, HH, WW, stride = conv_sets[i]  # num_filters, HH, WW, stride
            pad = int((HH - 1) / 2)
            self.conv_params[j] = {'stride':stride, 'pad':pad}
            H = 1 + (H + 2 * pad - HH) / stride
            W = 1 + (W + 2 * pad - WW) / stride
            # pad大法好
            C_p = int(C * self.p)
            self.params['P%d' % j] = np.random.randn(1, C_p, H, W)
            self.params['W%d' % j] = np.random.randn(F, C+C_p, HH, WW) / np.sqrt((C+C_p) * HH * WW / 2)
            self.params['b%d' % j] = np.zeros(F)

            if self.use_batchnorm:        # batchnorm初始化
                self.params['gamma%d' % j] = np.ones(F)  # (F * H * W)另一种batchnorm
                self.params['beta%d' % j] = np.zeros(F)  # (F * H * W)另一种batchnorm
            C = F  # 更新 C

            HH, WW, stride = pool_params[i]
            self.pool_params[j] = {'pool_height':HH, 'pool_width':WW, 'stride':stride}
            if HH > 1:
                H = 1 + (H - HH) / stride
                W = 1 + (W - WW) / stride
        # aff_layers initial
        aff_dims = [C * H * W] + aff_dim + [num_classes]
        for i in xrange(self.conv_layers, self.num_layers):  # 全连接初始化
            j = i + 1
            dim0, dim1 = aff_dims[i-self.conv_layers], aff_dims[j-self.conv_layers]
            self.params['W%d' % j] = np.random.randn(dim0, dim1) / np.sqrt(dim0 / 2)
            self.params['b%d' % j] = np.zeros(dim1)
            if self.use_batchnorm and j < self.num_layers:  # batchnorm初始化
                self.params['gamma%d' % j] = np.ones(dim1)
                self.params['beta%d' % j] = np.zeros(dim1)

        if not reset:
            params = io.loadmat('C:\\Users\\ZHB\\assignment2\\cs231n\\classifiers\\best_params.mat')  
            for j in xrange(1, self.num_layers+1):
                self.params['W%d' % j] = params['W%d' % j]
                self.params['b%d' % j] = params['b%d' % j][0]
                if self.use_batchnorm and j < self.num_layers:  # batchnorm初始化
                    self.params['gamma%d' % j] = params['gamma%d' % j]
                    self.params['beta%d' % j] = params['beta%d' % j]
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """forward"""
        conv_params, pool_params = self.conv_params, self.pool_params
        i, cache = 0, {}
        # convolution forward
        H = X.swapaxes(1, 0)
        for i in xrange(1, self.conv_layers+1):
            W, b = self.params['W%d' % i], self.params['b%d' % i]
            H, cache[i] = conv_forward_vector(H, W, b, conv_params[i])  # 卷积
        # affine forward
        H = H.swapaxes(1, 0)
        for i in xrange(self.conv_layers, self.num_layers):
            j = i + 1
            W, b = self.params['W%d' % j], self.params['b%d' % j]
            if j != self.num_layers:
                H, cache[j] = affine_relu_forward(H, W, b)  # 全连接
            else:
                H, cache[j] = affine_forward(H, W, b)  # 全连接
        scores = H
        # If test mode return early
        if y is None:
            return scores

        """backward"""
        loss, grads = 0.0, {}
        loss, dH = softmax_loss(scores, y)
        # affine backward
        for i in xrange(-self.num_layers, -self.conv_layers):
            j = -i
            W = self.params['W%d' % j]
            loss += 0.5 * self.reg * (np.sum(W*W))  # 加正则化loss
            if j == self.num_layers:
                dH, dW, db = affine_backward(dH, cache[j])  # 全连接
            else:
                dH, dW, db = affine_relu_backward(dH, cache[j])  # 全连接
            dW += self.reg * W  # 加正则化梯度
            grads['W%d' % j], grads['b%d' % j] = dW, db
        # convolution backward
        dH = dH.swapaxes(1, 0)
        for i in xrange(-self.conv_layers, 0):
            j = -i
            W = self.params['W%d' % j]
            loss += 0.5 * self.reg * (np.sum(W*W))  # 加正则化loss
            dH, dW, db = conv_backward_vector(dH, cache[j])  # 卷积
            dW += self.reg * W  # 加正则化梯度
            grads['W%d' % j], grads['b%d' % j] = dW, db
        # dx = dH.swapaxes(1, 0)

        return loss, grads
