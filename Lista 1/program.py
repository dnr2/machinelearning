'''
Cin UFPE - Aprendizagem de Maquina 2015.1
Lista de Exercicios #1
By: Danilo Neves Ribeiro dnr2@cin.ufpe.br

UCI Datasets:
- PROBLEM 1 - 
#UCI https://archive.ics.uci.edu/ml/datasets/Iris
#UCI https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
- PROBLEM 2 - 
#UCI https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
#UCI https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
'''

import numpy as np
import pandas as pd
import math

# ============ GLOBAL SETTINGS ============== #

#constants
attributes, classes = (0,1)
training, testing = (0,1)
EPS = 1e-9
k_values = [1,2,3,5,7,9,11,13,15]
train_data_ratio = 0.7;

#numpy settings
np.set_printoptions(threshold=15)
np.set_printoptions(precision=4)
np.random.seed(12314927)

# ============ UTILS ======================== #

def normalize(dataset):
  row_sums = dataset.sum(axis=1)
  new_matrix = dataset / row_sums[:, np.newaxis]
  return new_matrix

#read datasets from different file sources (headers must be removed)
#each dataset is split into attributes and classes (class column is assumed to be the last one)
#after that, each type of the dataset is split into training and and testing
def read_datasets(datafiles,normalize_attributes,class_last_column):
  #read data  
  datasets = [np.array(pd.read_csv(file,sep=',', header=None)) for file in datafiles]
  map(np.random.shuffle, datasets)
  
  #split datasets in two types: classes and attributes, index [0,1]  
  datasets = [ np.split(dataset, [dataset.shape[1]-1] if class_last_column[idx] else 1,axis=1 ) 
    for (idx,dataset) in list(enumerate(datasets)) ]    
  
  #data transformations
  if normalize_attributes :
    datasets = [ [normalize(dataset[attributes].astype(float)),dataset[classes]] 
      for dataset in datasets ];          
  
  #splits data set in training and testing data, index [0,1]
  datasets = [ [ np.split(dataset[type], [int(train_data_ratio * dataset[type].shape[0])]) 
    for type in [attributes,classes]]  for dataset in datasets]  
  return datasets

# ============ DISTANCES ==================== #
  
def euclidian_dist(vec1, vec2):  
  sum = 0
  if len(vec1) != len(vec2):
    return float('nan')
  for i in range(len(vec1)):
    sum = sum + math.pow(vec1[i] - vec2[i] , 2)
  return math.sqrt(sum)

#TODO!!! how to integrate?
def vdm_distance(vec1,vec2,dataset):
  sum = 0
  if len(vec1) != len(vec2):
    return float('nan')
  for i in range(len(vec1)):
    print i
  return 0

# ============ ALGORITHMS =================== #

def k_nn_predict_class(query, k_value, dataset, weighted, dist_func):  
  #set first instance to be the base case
  ini_dist = dist_func(dataset[attributes][training][0,:] ,query)
  ini_class = dataset[classes][training][0,0]
  k_nearest = []
    
  #TODO improve performance here using heap and iterators over dataset
  #find k-nearest_neighbours 
  for row in range(dataset[attributes][training].shape[0]):    
    cur_dist = dist_func(dataset[attributes][training][row,:],query)    
    cur_class = dataset[classes][training][row,0]                    
    if len( k_nearest ) == k_value:
      if cur_dist < k_nearest[0][0]:       
        k_nearest[0] = ( cur_dist, cur_class )
    else :
      k_nearest.append((cur_dist, cur_class)) 
    k_nearest = sorted( k_nearest, key= lambda x : x[0], reverse=True )
    
  #counting for each class (with or without weight)
  counting = {}
  for instance in k_nearest:      
    counting.setdefault(instance[1], 0)
    if not weighted:
      counting[instance[1]] += 1
    else:      
      #weight avoiding division by zero
      weight = 1.0 / (math.pow(float(instance[0]),2.0) + EPS)
      counting[instance[1]] += weight
  
  #get predicted_class
  predicted_class = None
  best_counting = 0
  for key in counting:
    if predicted_class is None or best_counting < counting[key]:
      predicted_class = key
      best_counting = counting[key]
  
  return predicted_class
  

# ============ SOLUTION FOR PROBLEMS ======== #

def solve_knn(datasets,dist_func):
  #run training with different configurations
  for dataset in datasets:     
    for k_nn_weighted in [False,True]:      
      print "k_nn_weighted ", k_nn_weighted
      for k_value in k_values:
        print "k_value ", k_value
        accuracy_sum = 0        
        for query_idx in range(dataset[attributes][testing].shape[0]):
          query = dataset[attributes][testing][query_idx,:]        
          real_class = dataset[classes][testing][query_idx,0]
          predicted_class = k_nn_predict_class(query, k_value, dataset, k_nn_weighted, dist_func)        
          if predicted_class == real_class:
            accuracy_sum += 1
        dataset_accuracy = float(accuracy_sum) / float(dataset[attributes][testing].shape[0])
        print dataset_accuracy

#using dataset iris and transfusion from UCI, headers were removed
def solve_problem1():
  #set parameters      
  datafiles = ["iris.data.txt", "transfusion.data.txt"]      
  normalize_attributes = True
  class_last_column = [True,True]
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,normalize_attributes,class_last_column)    
  solve_knn(datasets,euclidian_dist)

#using datasets Tic-Tac-Toe Endgame and Congressional Voting from UCI
def solve_problem2():
  #set parameters  
  datafiles = ["tic-tac-toe.data.txt", "house-votes-84.data.txt"]    
  normalize_attributes = False
  class_last_column = [True,False]
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,normalize_attributes,class_last_column)    
  solve_knn(datasets,vdm_distance)
  
def main():
  solve_problem1()  
  
if __name__ == "__main__":
  main()