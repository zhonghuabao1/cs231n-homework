# -*- coding: utf-8 -*-
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from scipy import io

class ConvNet(object):
    def __init__(self, input_dim=(3, 32, 32),
                 conv_sets=[(32,3,3,2), (32,3,3,2)],
                 pool_params=[(2,2,2), (2,2,2)],
                 aff_dim=[100, 100], num_classes=10,
                 reg=0.0, use_batchnorm=True,
                 reset=False, dtype=np.float32):
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.params, self.conv_params, self.pool_params = {}, {}, {}
        self.conv_layers = len(conv_sets)
        C, H, W = input_dim
        for i in xrange(self.conv_layers):  # 卷积核初始化
            j = i + 1
            F, HH, WW, stride = conv_sets[i]  # num_filters, HH, WW, stride
            pad = (HH - 1) / 2
            self.conv_params[j] = {'stride':stride, 'pad':pad}
            H = 1 + ( H + 2 * pad - HH) / stride
            W = 1 + ( W + 2 * pad - WW) / stride
            self.params['W%d' % j] = np.random.randn(F, C, HH, WW) / np.sqrt(C * HH * WW / 2)
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

        self.num_layers = self.conv_layers + len(aff_dim) + 1
        aff_dims = [C * H * W] + aff_dim + [num_classes]
        for i in xrange(self.conv_layers, self.num_layers):  # 全连接初始化
            j = i + 1
            dim0, dim1 = aff_dims[i-self.conv_layers], aff_dims[j-self.conv_layers]
            self.params['W%d' % j] = np.random.randn(dim0, dim1) / np.sqrt(dim0 / 2)
            self.params['b%d' % j] = np.zeros(dim1)
            if self.use_batchnorm and j < self.num_layers:  # batchnorm初始化
                self.params['gamma%d' % j] = np.ones(dim1)
                self.params['beta%d' % j] = np.zeros(dim1)

        if reset == False:
            params = io.loadmat('C:\\Users\\ZHB\\assignment2\\cs231n\\classifiers\\best_params.mat')  
            for j in xrange(1, self.num_layers+1):
                self.params['W%d' % j] = params['W%d' % j]
                self.params['b%d' % j] = params['b%d' % j][0]
                if self.use_batchnorm and j < self.num_layers:  # batchnorm初始化
                    self.params['gamma%d' % j] = params['gamma%d' % j]
                    self.params['beta%d' % j] = params['beta%d' % j]
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
            
    def data_extend(self, X, y):
        if mode == 'train' and False: 
            i = 4
            N, C, H, W = X.shape
            X_new = np.zeros([N+4*i, C, H, W]).astype(np.float32)
            y_new = np.zeros(N+4*i).astype(np.int32)
            X_new[:N] = X
            y_new[:N] = y
            X_new[N:N+i, :, 1:, :] = X[:i, :, :-1, :] 
            X_new[N+i:N+2*i, :, :-1,:] = X[i:2*i, :, 1:,:] 
            X_new[N+2*i:N+3*i,:,:,:-1] = X[2*i:3*i,:,:,1:] 
            X_new[N+3*i:N+4*i, :, :, 1:] = X[3*i:4*i, :, :, :-1] 
            y_new[N:N+4*i] = y[:4*i]
            X, y = X_new, y_new
        i = 2
        j = 2 * i
        N, C, H, W = X.shape
        X_pad = np.zeros([N, C, H+j, W+j]).astype(np.float32)
        X_pad[:,:,i:H+i,i:W+i] = X
        X_new = np.zeros([4*N, C, H, W]).astype(np.float32)
        y_new = np.zeros(4*N).astype(np.int32)
        X_new[:N] = X
        y_new[:N] = y
        X_new[N:2*N] = X_pad[:,:,:H,:W]
        y_new[N:2*N] = y
        X_new[2*N:3*N] = X_pad[:,:,j:,j:]
        y_new[2*N:3*N] = y
        X_new[3*N:4*N] = np.fliplr(X.swapaxes(1,3)).swapaxes(1,3)
        y_new[3*N:4*N] = y
        return X_new, y_new
            
    def loss(self, X, y=None):
        #################################### forward
        mode = 'test' if y is None else 'train'
        if mode == 'train': 
            i = 16
            N, C, H, W = X.shape
            X_new = np.zeros([N+4*i, C, H, W]).astype(np.float32)
            y_new = np.zeros(N+4*i).astype(np.int32)
            X_new[:N] = X
            y_new[:N] = y
            X_new[N:N+i, :, 1:, :] = X[:i, :, :-1, :] 
            X_new[N+i:N+2*i, :, :-1,:] = X[i:2*i, :, 1:,:] 
            X_new[N+2*i:N+3*i,:,:,:-1] = X[2*i:3*i,:,:,1:] 
            X_new[N+3*i:N+4*i, :, :, 1:] = X[3*i:4*i, :, :, :-1] 
            y_new[N:N+4*i] = y[:4*i]
            X, y = X_new, y_new
        conv_params, pool_params = self.conv_params, self.pool_params
        H, cache = X, {}

        if self.use_batchnorm:
            bn_params = {}
            for j in xrange(1, self.num_layers):
                bn_params[j] = {'mode': mode}
                W, b = self.params['W%d' % j], self.params['b%d' % j]
                gamma, beta = self.params['gamma%d' % j], self.params['beta%d' % j]
                if j <= self.conv_layers:
                    H, cache[j] = conv_bn_relu_pool_forward(H, W, b, conv_params[j],
                                                           gamma, beta, bn_params[j], pool_params[j])  # 卷积
                else: 
                    H, cache[j] = aff_bn_relu_forward(H, W, b, gamma, beta, bn_params[j])  # 全连接
        else:
            for j in xrange(1, self.num_layers):
                W, b = self.params['W%d' % j], self.params['b%d' % j]
                if j <= self.conv_layers:
                    H, cache[j] = conv_relu_pool_forward(H, W, b, conv_params[j], pool_params[j])  # 卷积
                else: H, cache[j] = affine_relu_forward(H, W, b)  # 全连接
        j +=1
        scores, cache[j] = affine_forward(H, self.params['W%d' % j], self.params['b%d' % j])
        # If test mode return early
        if y is None:
            return scores
        #################################### backward
        loss, grads = 0.0, {}
        loss, dH = softmax_loss(scores, y)
        j = self.num_layers
        W = self.params['W%d' % j]
        loss += 0.5 * self.reg * (np.sum(W*W))  # 加正则化loss
        dH, dW, db = affine_backward(dH, cache[j])
        dW += self.reg * W  # 加正则化梯度
        grads['W%d' % j], grads['b%d' % j] = dW, db
        if self.use_batchnorm:
            for i in xrange(1 - self.num_layers, 0):
                j = -i
                W = self.params['W%d' % j]
                loss += 0.5 * self.reg * (np.sum(W*W))  # 加正则化loss
                if j > self.conv_layers:
                    dgamma, dbeta, dH, dW, db = aff_bn_relu_backward(dH, cache[j])  # 全连接
                else:
                    dgamma, dbeta, dH, dW, db = conv_bn_relu_pool_backward(dH, cache[j])  # 卷积
                dW += self.reg * W  # 加正则化梯度
                grads['gamma%d' % j], grads['beta%d' % j] = dgamma, dbeta
                grads['W%d' % j], grads['b%d' % j] = dW, db
        else:
            for i in xrange(1 - self.num_layers, 0):
                j = -i
                W = self.params['W%d' % j]
                loss += 0.5 * self.reg * (np.sum(W*W))  # 加正则化loss
                if j > self.conv_layers:
                    dH, dW, db = affine_relu_backward(dH, cache[j])  # 全连接
                else:
                    dH, dW, db = conv_relu_pool_backward(dH, cache[j])  # 卷积
                dW += self.reg * W  # 加正则化梯度
                grads['W%d' % j], grads['b%d' % j] = dW, db
        return loss, grads

