# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:45:03 2018

@author: Isnanda
"""

import random
import numpy as np
import time

def getTarget(data_transpose_list, perc):
    value_target = []
    for x in range(len(data_transpose_list)):
        p = np.percentile(data_transpose_list[x], perc)
        value_target.append(p)
        
    return value_target

""" ======================================================================= """

def init_gen(data, population):
    sample_gen = []
    for x in range(population):
        sample_gen.append(random.choice(data))
        
    return sample_gen

""" ======================================================================= """

def calc_fitness(sample_gen, value_target):
    fitness = np.zeros(len(sample_gen))
    nilai_persen = 0
    nilai_persen_akhir = []
    nilai_akhir = 0
    for x in range(len(sample_gen)):
        nilai_persen = 0
        nilai_akhir = 0
        for z in range(len(sample_gen[x])):
            if value_target[z] < sample_gen[x][z]:
                nilai_persen += (value_target[z]/sample_gen[x][z])
            else:
                nilai_persen += (sample_gen[x][z]/value_target[z])
        nilai_akhir = 1/(1+(nilai_persen/len(value_target)))
        #print("Nilai Persen : ", (nilai_persen/len(value_target)))
        #print("Nilai Akhir : ",nilai_akhir)
        #time.sleep(5)
        nilai_persen_akhir.append(nilai_akhir)
        
    return nilai_persen_akhir

""" ======================================================================= """

def selection(sample_gen, fitness_value):
    for index in range(1, len(fitness_value)):
        currentFitness = fitness_value[index]
        currentSample = sample_gen[index]
        position = index
        
        while position > 0 and fitness_value[position -1]>currentFitness:
            fitness_value[position] = fitness_value[position - 1]
            sample_gen[position] = sample_gen[position - 1]
            position = position - 1
        
        fitness_value[position] = currentFitness
        sample_gen[position] = currentSample
    
    selection_gen = sample_gen[:int(0.4 * len(sample_gen))]
    selection_fitness = fitness_value[:int(0.4 * len(fitness_value))]
    
    return selection_gen, selection_fitness

""" ======================================================================= """

def crossover(selection_gen, population, length):
    offspring = []
    for x in range(round((population - len(selection_gen))/2)):
        parent1 = random.choice(selection_gen)
        #print("Parent - 1 : ",parent1)
        parent2 = random.choice(selection_gen)
        #print("Parent - 2 : ",parent2)
        split = random.randint(0, length)
        #print("Nilai Split : ",split)
        child1 = parent1[0:split] + parent2[split:length]
        child2 = parent2[0:split] + parent1[split:length]
        #print("Child 1 : ",child1," == Child 2 : ",child2)
    
        offspring.append(child1)
        offspring.append(child2)

    selection_gen.extend(offspring)
    
    return selection_gen

""" ======================================================================= """

def mutation(selection_gen, data):
    for x in range(len(selection_gen)):
        ex = selection_gen[x]
        for idx, param in enumerate(ex):
            if random.uniform(0.0, 1.0) <= 0.1:
                ex[idx] = random.choice(data[idx])
                #print("Nilai Gen Terpilih : ", selection_gen[x])
        selection_gen[x] = ex
        #print("Nilai Gen Setelah Mutasi : ", selection_gen[x])
    
    return selection_gen

""" ======================================================================= """

def buildGA(data_latih, perc):
    
    """
    KETERANGAN 
    value_target -> list (1 x 6)
    sample_gen -> list (20 x 6)
    
    
    """
    
    population = 5
    generations = 10000
    
    """
    PREPROCESS DATA
    """
    np_data = np.array(data_latih)
    data_transpose = np_data.T
    data_transpose_list = []
    for x in range(len(data_transpose)):
        data_transpose_list.append(data_transpose[x])
    
    value_target = getTarget(data_transpose_list, perc)
    #print("Nilai Value Target : ", value_target)
    #time.sleep(15) # Mendapatkan Nilai Fungsi Objektif
    sample_gen = init_gen(data_latih, population)
    #print("Nilai Sample Gen : ", sample_gen)
    #time.sleep(15)# Mendapatkan Populasi Awal
    
    for generation in range(generations):
        fitness = calc_fitness(sample_gen, value_target)
        #print("Nilai Fitness : ", fitness)
        sample_gen, selection_fitness = selection(sample_gen, fitness)
        
        if any(fitness_result <= 0.5 for fitness_result in fitness):
            print("Target : ", value_target)
            print("Centroids : ", sample_gen[0])
            print("Fitness : ", fitness[0])
            print("Generation : ", generation)
            break
        
        sample_gen = crossover(sample_gen, population, len(value_target))
        sample_gen = mutation(sample_gen, data_transpose_list)
    
    return sample_gen[0], value_target