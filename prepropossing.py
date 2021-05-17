#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import wfdb
from tqdm import tqdm
from scipy.signal import resample
import matplotlib.pyplot as plt

def getdata(datapath):
    pac= ['a','J','A','S']
    pvc = ['V','r']
    mitdb=[]
    mitdb_a=[]
    mitdb_v=[]
    sample = wfdb.rdsamp(datapath)
    ln = (sample[0].shape[0])//360
    s = []
    s.append(resample(sample[0][:ln*360,0],ln*400).astype(np.float16))#
    s.append(resample(sample[0][:ln*360,1],ln*400).astype(np.float16))#
    ann = wfdb.rdann('data//118','atr')
    beats_a = ann.sample[np.isin(ann.symbol,pac)]*400//360
    beats_v = ann.sample[np.isin(ann.symbol,pvc)]*400//360
    mitdb.append(s)
    mitdb_a.append(beats_a)
    mitdb_v.append(beats_v)
    return mitdb,mitdb_a,mitdb_v

def seg(mitdb,mitdb_a,mitdb_v):
    data_all = []
    ref_all = []
    labels=[]
    label = np.zeros((len(mitdb[0][0])))
    for j in mitdb_a:
        label[j] = 1
    for j in mitdb_v:
        label[j] = 2
    labels.append(label)
    length = len(mitdb[0][0])
    k=0#MLII
    for s in mitdb_a[0]:
        fr = max(80,s-3600)
        to = min(s+3600,length-40)
        d_s = mitdb[0][k][fr:to]
        r_s = labels[0][fr:to]
        if to-fr < 4000:
            z1 = np.zeros((4000))
            z1[:to-fr] = d_s
            d_s = z1
        data_all.append(d_s.astype(np.float16))
        ref_all.append(r_s.astype(np.int8))
    for j in mitdb_v[0]:
        s = j//8*8
        fr = max(80,s-3600)
        to = min(s+3600,length-40)
        d_s = mitdb[0][k][fr:to]
        r_s = labels[0][fr:to]
        if to-fr < 4000:
            z1 = np.zeros((4000))
            z1[:to-fr] = d_s
            d_s = z1
        data_all.append(d_s.astype(np.float16))
        ref_all.append(r_s.astype(np.int8))
    return data_all,ref_all

def y_gen1(label,smooth=0.03):##S
    a = np.where(label==1)[0]
    y = np.zeros((125,1))
    for i in a:
        w1 = i//32
        y[w1][0] = 1-smooth
    return y

def y_gen2(label,smooth=0):##V
    v = np.where(label==2)[0]
    y = np.zeros((125,1))
    for i in v:
        w1 = i//32
        y[w1][0] = 1-smooth
    return y

def data_gen(ID,batch=500,aug=False,gen=y_gen2):
    loc = 0
    while 1:
        samples = np.zeros((batch,4000))
        labels = np.zeros((batch,125,1))
        for i in range(batch):
            ecg = np.copy(data[ID[i+loc]])
            if aug:
                ecg *= 0.1*np.random.randint(8,13)
                ecg += np.random.normal(0,0.001*np.random.randint(0,10),len(ecg))
                ecg += 0.01*np.random.randint(-20,21)
            fr = np.random.randint(len(ecg)-3999)
            samples[i] = ecg[fr:fr+4000]
            labels[i] = gen(ref[ID[i+loc]][fr:fr+4000],smooth=0)
        loc += batch
        if loc+batch>=len(ID):
            loc=0
            np.random.shuffle(ID)
        yield [samples,labels],np.zeros((batch))

