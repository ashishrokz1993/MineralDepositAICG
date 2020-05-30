'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module contains the Adam optimizer classes and functions
'''
import tensorflow as tf
import numpy as np


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out
def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)]) for (v, grad) in zip(var_list, grads)])

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)

class Adam(object):
    #See https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer for adam optimizer details
    def __init__(self, var_list, *,beta1=0.9, beta2=0.999,epsilon = 1e-08, scalegrad=True):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon= epsilon
        self.scalgrad = scalegrad
        size = sum(numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size,'float32')
        self.t =0
        self.setFlat = SetFromFlat(var_list)
        self.getFlat = GetFlat(var_list)

    def update(self,localg, stepsize):
        localg = localg.astype('float32')
        globalg = np.copy(localg)
        self.t+=1
        a = stepsize*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        self.m = self.beta1*self.m+(1-self.beta1)*globalg
        self.v = self.beta2*self.v+(1-self.beta2)*(globalg*globalg)
        step = (-a)*self.m/(np.sqrt(self.v)+self.epsilon)
        self.setFlat(self.getFlat()+step)


    def sync(self):
        theta = self.getFlat()
        self.setFlat(theta)


