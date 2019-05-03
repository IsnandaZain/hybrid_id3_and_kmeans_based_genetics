# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:18:22 2018

@author: Isnanda Muhammad Zain
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import time

def euc_distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def build(data_training, centroids_awal):
    """
    BUILD KMeans Algorithm
    data => data final yang siap untuk diolah
    centroids_awal => centroids awal untuk iterasi
    """
    columns = len(data_training[0])
    centroids_data = np.array(centroids_awal, dtype=np.float32)
    centroids_data_lama = np.zeros(centroids_data.shape)
    clusters = np.zeros(len(data_training))
    error = euc_distance(centroids_awal, centroids_data_lama, None)
    
    iterasi = 0
    while error != 0:
        iterasi += 1
        print("Nilai Centroids Sekarang : ", centroids_data)
        for i in range(len(data_training)):
            distances = euc_distance(data_training[i], centroids_data)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        
        centroids_data_lama = deepcopy(centroids_data)
        for i in range(len(centroids_data)):
            points = [data_training[j] for j in range(len(data_training)) if clusters[j] == i]
            centroids_data[i] = np.mean(points, axis=0)
        error = euc_distance(centroids_data, centroids_data_lama, None)
        time.sleep(2)
        
        np_data = np.array(data_training)
        from sklearn.metrics import silhouette_score
        np_target = np.array(clusters)
        s = silhouette_score(np_data, np_target)
        print("Nilai Silhouette Index : ", s)
        
        time.sleep(5)
    
    print("Jumlah Iterasi : ", iterasi)
    print("Nilai Centroids Sekarang : ", centroids_data)
    print("Nilai Centroids Sebelumnya : ", centroids_data_lama)
    return clusters