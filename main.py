""" ====================================================================== """
import id3_test as ID3
import k_means_test as KMeans
import used_ga as GA
import pandas as pd
import pprint
from copy import deepcopy
import numpy as np
import time
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold

print("============= WELCOME TO MAIN PROGRAM ===========\n\n")

def predict(inst, tree):
    for nodes in tree.keys():
        prediction = 0
        try:
            value = inst[nodes]
            tree = tree[nodes][value]
            print("Nilai Value : ", value, " == ", nodes)
        
            if type(tree) is dict:
                prediction = predict(inst, tree)
            else:
                prediction = tree
                break;
        except KeyError:
            prediction = 99
            #raise predict()
            
    return prediction

""" ====================================================================== """

def preprocess(data):
    atribut = data.columns.values.tolist()
    """
    for x in range(len(columns_index)):
        for i in range(len(data.columns.values)):
            if str(atribut[i]) == columns_index[x]:
                data.drop(columns = atribut[i], inplace=True)
    """
    
    data_final = []
    atribut = data.columns.values
    for j in range(len(data)):
        data_final.append([])
        for z in range(len(data.columns.values)):
            data_final[j].append(data[data.columns.values[z]][j])
    
    return data_final, atribut, data

"""
PROSES MEMBACA DAN MEREDUPLIKASI DATA
"""
filename = input("Nama File : ")
data = pd.read_csv(filename)
data_process = deepcopy(data)
#data_process.drop(columns = data_process.columns.values[0], inplace=True)
print("Thankss for your Information, Wait for me process them")
data_latih, atribut, data_baru = preprocess(data_process)
time.sleep(2)
print("PROCESSING DONE ... 10%\n")
    
"""
PROSES K-MEANS CLUSTERING
"""
jml_centroids = int(input('Jumlah centroids? : '))

perc = []
jml_ga = jml_centroids+1
for x in range(1,jml_ga):
    perc.append(round((x/jml_ga)*100))
print("Nilai Presentase (%) : ", perc)


centroids_awal = [[1, 1, 6, 20, 1, 1], [1, 1, 7, 20, 1, 1], [2, 2, 8, 21, 1, 2]]
#ALGORITMA GENETIKA

centroids_awal = []

np_data = np.array(data_latih)
data_transpose = np_data.T
data_tranpose_list = []
"""
for x in range(jml_centroids):
    centroids_awal.append([])
    for z in range(len(data_process.columns.values)):
        centroids_awal[x].append(random.choice(data_transpose[z]))
    print("Nilai Centroids Awal : " , centroids_awal[x])
"""
for x in range(jml_centroids):
    centroidsawal, nilai_target = GA.buildGA(data_latih, perc[x])
    time.sleep(2)
    centroids_awal.append(centroidsawal)

cluster_result = KMeans.build(data_latih, centroids_awal)

from sklearn.metrics import silhouette_score
np_target = np.array(cluster_result)
s = silhouette_score(np_data, np_target)
print("Nilai Silhouette Index : ", s)

"""
PROSES ID3
"""
data['Class'] = pd.Series(cluster_result, index=data.index)
data.to_csv('DataHasil_Cluster.csv')
atribut = data.columns.values.tolist()

"""
BUILD TREE GENERAL
""""
data_baru = pd.read_csv('DataHasil_Cluster.csv')
data_final, atribut, data_baru = preprocess(data_baru)
data_baru.drop(columns = data_baru.columns.values[0], inplace=True)

tree_result = ID3.buildTree(data_baru)
pprint.pprint(tree_result)



nilai_accuracy = []
X_baru = np.array(data_latih)
Y_baru = np.array(cluster_result)
kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(X_baru)
print(kf)

for train_index, test_index in kf.split(X_baru):
    print("TRAIN : ", train_index, " TEST : ", test_index)
    X_train, X_test = X_baru[train_index], X_baru[test_index]
    Y_train, Y_test = Y_baru[train_index], Y_baru[test_index]
    
    data_process = pd.DataFrame(np.column_stack([X_train, Y_train]), columns=atribut)
    tree_result = ID3.buildTree(data_process)
    
    #pprint.pprint(tree_result)
    #print(X_train)
    #print(Y_train)
    
    Y_pred = []
    for x in range(len(X_test)):
        data_testing = dict(zip(atribut[:-1], X_test[x]))
        #print(data_testing)
        inst = pd.Series(data_testing)
        prediction = predict(inst, tree_result)
        Y_pred.append(prediction)
        print("Y_test : ", Y_test[x], " == ", Y_pred[x])
    
    print("Accuracy is : ", accuracy_score(Y_test, Y_pred) * 100)
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    nilai_accuracy.append(accuracy * 100)
    time.sleep(5)

for x in range(len(nilai_accuracy)):
    print("Nilai Akurasi ke - ",x+1, " = ", nilai_accuracy[x])
np_accuracy = np.array(nilai_accuracy)
print("Nilai rata-rata Akurasi = ", np.mean(np_accuracy), "%")

""" ======================================================================= """


data_cluster = pd.read_csv("Dataset.csv")
data_cluster.drop(columns = data_cluster.columns.values[0], inplace=True)
data_cluster.drop(columns = data_cluster.columns.values[0], inplace=True)
data_cluster["Cluster"] = pd.Series(cluster_result, index=data.index)

atribut = data.columns.values.tolist()
    
data_cluster_satu = []
cs, cd, ct = 0 , 0 ,0
data_cluster_dua = []
data_cluster_tiga = []
for j in range(len(data_cluster)):
    if data_cluster[data_cluster.columns.values[-1]][j] == 0:
        data_cluster_satu.append([])
        for z in range(len(data_cluster.columns.values)):
            data_cluster_satu[cs].append(data_cluster[data_cluster.columns.values[z]][j])
        cs += 1
    elif data_cluster[data_cluster.columns.values[-1]][j] == 1:
        data_cluster_dua.append([])
        for z in range(len(data_cluster.columns.values)):
            data_cluster_dua[cd].append(data_cluster[data_cluster.columns.values[z]][j])
        cd += 1
    else:
        data_cluster_tiga.append([])
        for z in range(len(data_cluster.columns.values)):
            data_cluster_tiga[ct].append(data_cluster[data_cluster.columns.values[z]][j])
        ct += 1
    
""" ======================================================================= """