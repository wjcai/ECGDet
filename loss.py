#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import backend as K

def robust_loss(args):
    y_true = args[1]
    outputs = args[0]
    loss = 0
    m = K.shape(outputs)[0]
    mf = K.cast(m, K.dtype(outputs))

    zero = tf.zeros((m,1))
    zero = tf.cast(zero,y_true.dtype)
    object_mask = y_true[..., 0]#
    mask = object_mask[...,2:-2]+object_mask[...,1:-3]+object_mask[...,0:-4]+object_mask[...,3:-1]+object_mask[...,4:]
    ignore_mask = K.concatenate([zero,zero,mask,zero,zero])<1
    ignore_mask = K.cast(ignore_mask,y_true.dtype)
    confidence_loss =7*K.binary_crossentropy(object_mask, outputs[...,0]) + ignore_mask *  K.binary_crossentropy(object_mask, outputs[...,0])
    confidence_loss = K.sum(confidence_loss,axis=-1)
    loss += confidence_loss

    return loss




