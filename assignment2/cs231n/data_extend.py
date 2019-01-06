# -*- coding: utf-8 -*-

import numpy as np

def data_extend(X, y):
    X_new, y_new = X.copy(), y.copy()
    # 水平镜面翻转
    X_extend = X[:,:,:,::-1]
    X_new = np.concatenate((X_new, X_extend), axis=0)  # 追加
    y_new = np.concatenate((y_new, y), axis=0)
    # 左上角平移
    X_extend = np.zeros_like(X)
    X_extend[:,:,:-1,:-1] = X[:,:,1:,1:]
    X_new = np.concatenate((X_new, X_extend), axis=0)  # 追加
    y_new = np.concatenate((y_new, y), axis=0)
    # 右下角平移
    X_extend = np.zeros_like(X)
    X_extend[:,:,1:,1:] = X[:,:,:-1,:-1]
    X_new = np.concatenate((X_new, X_extend), axis=0)  # 追加
    y_new = np.concatenate((y_new, y), axis=0)

    return X_new, y_new
