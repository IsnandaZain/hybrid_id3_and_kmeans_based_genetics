# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:45:03 2018

@author: Isnanda
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import time
import math

def entropy_target(data):
    class_target = data.keys()[-1]
    entropy = 0
    
    nilai_data = data[class_target].unique()
    for n in nilai_data:
        jumlah_data_atribut = data[class_target].value_counts()[n] / len(data[class_target])
        entropy += -jumlah_data_atribut * math.log(jumlah_data_atribut,3)
    
    print("Nilai Entropy - ", class_target, " = ", entropy)        
    return entropy

def entropy_atribut(data, atribut):
    class_target = data.keys()[-1]
    nilai_kelas_target = data[class_target].unique()
    class_atribut = data[atribut].unique()
    entropy_result = 0
    
    for n in class_atribut:
        entropy = 0
        for target in nilai_kelas_target:
            total_kejadian = len(data[atribut][data[atribut] == n])
            total_kejadian_target = len(data[atribut][data[atribut] == n][data[class_target] == target])
            temp = total_kejadian_target / total_kejadian
            if temp == 0 or temp == 1:
                entropy += 0
            else:
                entropy += -temp * math.log(temp,3)
        
        temp_result = total_kejadian / len(data)
        entropy_result += -temp_result * entropy
    return abs(entropy_result)

def info_gain(data):
    entropy_attr = []
    ig = []
    for key in data.keys()[:-1]:
        ig.append(entropy_target(data) - entropy_atribut(data, key))

    return data.keys()[:-1][np.argmax(ig)]

def get_leaf(data, node, value):
    return data[data[node] == value].reset_index(drop=True)

def buildTree(data, tree=None):
    class_target = data.keys()[-1]
    node = info_gain(data)
    atribut = np.unique(data[node])
    if tree is None:
        tree = {}
        tree[node] = {}
        
    for n in atribut:
        leaf = get_leaf(data, node, n)
        nilai, jumlah = np.unique(leaf[''.join(class_target)], return_counts=True)
        
        if len(jumlah) == 1:
            tree[node][n] = nilai[0]
        else:
            tree[node][n] = buildTree(leaf)
            
    return tree