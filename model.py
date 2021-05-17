#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import Input
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from loss import *

def mish(inputs):
    return inputs * K.tanh(K.softplus(inputs))

def conv_block(x,filters,kernel_size,strides,padding):
    x = layers.Conv1D(filters,kernel_size,strides=strides,padding=padding)(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Activation(sin)(x)
    return x
    
def se_block(x,filters,kernel_size,stride=1,change=False):
    x_short = x
    x = conv_block(x,filters,kernel_size,strides=1,padding='same')
    x = layers.Activation(mish)(x)
    x = layers.Dropout(0.2)(x)
    x = conv_block(x,filters,kernel_size,strides=stride,padding='same')
    se = layers.GlobalAveragePooling1D()(x)
    se= layers.Dense(filters//16,activation=mish)(se)
    se= layers.Dense(filters,activation='sigmoid')(se)
    se = layers.Reshape((1,filters))(se)
    x = layers.Multiply()([se,x])
    if stride>1 or change:
        x_short = conv_block(x_short,filters,1,strides=stride,padding='same')
    x = layers.Add()([x,x_short])
    x = layers.Activation(mish)(x)
    return x

def model_build():
    inputs = Input(shape=(None,), dtype='float32')
    x = layers.Reshape((-1,4))(inputs)
    x = layers.Conv1D(32, 16, strides=1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(mish)(x)
    x = se_block(x,32,11,stride=1)
    x = se_block(x,32,11,stride=1)
    x = se_block(x,32,11,stride=2)#se_block3
    x = se_block(x,64,9,stride=1,change=True)#se_block4
    x = se_block(x,64,9,stride=1)
    x = se_block(x,64,9,stride=2)#se_block6
    x = se_block(x,128,7,stride=1,change=True)#se_block7
    x = se_block(x,128,7,stride=1)
    x = se_block(x,128,7,stride=2)#se_block9
    x = se_block(x,128,5,stride=1,change=True)#se_block10
    x = se_block(x,128,5,stride=1)
    x = se_block(x,128,5,stride=1)
    out = layers.Dense(1,activation='sigmoid')(x)
    backbone = Model(inputs,out)
    y_true = Input(shape=(None,1))
    loss_input = [backbone.output, y_true]
    model_loss = Lambda(robust_loss, output_shape=(1,), name='robust_loss')(loss_input)
    model = Model(inputs=[backbone.input, y_true], outputs=model_loss)
    model.compile(optimizer=Adam(lr=0.001,clipnorm=1.),loss={'robust_loss': lambda y_true, y_pred: y_pred})
    #model.summary()
    #plot_model(model, to_file='NetStruct.png', show_shapes=True)
    return model


