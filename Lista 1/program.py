'''
Cin UFPE - Aprendizagem de Maquina 2015.1
Lista de exercicios 1
By: Danilo Neves Ribeiro dnr2@cin.ufpe.br
'''

import numpy as np
import pandas as pd

# ============ GLOBAL SETTINGS ============== #

np.set_printoptions(threshold=15)
np.set_printoptions(precision=4)
attributes, classes = (0,1)
training, testing = (0,1)

# ============ UTILS ======================== #

def normalize(dataset):
  row_sums = dataset.sum(axis=1)
  new_matrix = dataset / row_sums[:, np.newaxis]
  return new_matrix

#read datasets from different file sources (headers must be removed)
#each dataset is split into attribues and classes (classes colum is assumed to be the last one)
#after that each dataset is split into training and and testing
def read_datasets( datafiles, normalize_attributes, train_data_ratio):
  #read data  
  datasets = [ np.array( pd.read_csv(file,sep=',', header=None)) for file in datafiles]
  
  map(np.random.shuffle, datasets)
  
  #split datasets in two types: classes and attributes, index [0,1]
  datasets = [ np.split(dataset,[dataset.shape[1]-1],axis=1 ) for dataset in datasets ]  
  
  #data transformations
  if normalize_attributes :
    datasets = [ [normalize(dataset[attributes].astype(float)),dataset[classes]] 
      for dataset in datasets ];        
  
  #splits data set in training and testing data, index [0,1]
  datasets = [ [ np.split(dataset[type], [int(train_data_ratio * dataset[type].shape[0])]) 
    for type in [attributes,classes]]  for dataset in datasets]  
  return datasets

def euclidian_dist(vec1, vec2):
  sum = 0
  if len(vec1) != len(vec2):
    return float('nan');  
  for i in range(len(vec1)):
    sum = sum + math.pow(vec1[i] - vec2[i] , 2)
  return math.sqrt(sum)
  
def np_euclidian_dis(vec1, vec2):
  return numpy.linalg.norm(vec1-vec2)

# ============ ALGORITHMS =================== #

def k_nn_weighted(query, k_value, dataset):  
  #set first instance to be the base case
  ini_dist = euclidian_dist(dataset[attributes][training][0,:] ,query)
  ini_class = dataset[0,:][classes][training]
  k_nearest = [( ini_dist, ini_class)]
  
  #find k-nearest_neighbours 
  for instance in dataset:
    sorted( k_nearest, key= lambda x : x[0], reverse=True )
    cur_dist = euclidian_dist(instance[attributes][training],query)
    cur_class = instance[classes][training]
    if len( k_nearest ) == k_value and cur_dist < k_nearest[0][0] :
      k_nearest[0] = ( cur_dist, cur_class )
    else :
      k_nearest.append( ( cur_dist, cur_class ) ) 
  counting = {}
  
  #TODO!!!!!!!!!!!!!!!!!!!!!!!
  return k_nearest[0][1]
  

# ============ SOLUTION PROBLEM 1 =========== #

#using dataset iris and transfusion from UCI, headers were removed
def solve_problem1():
  #set parameters  
  datafiles = ["iris.data.txt", "transfusion.data.txt"]  
  k_values = [1,2,3,5,7,9,11,13,15]
  train_data_ratio = 0.7;
  normalize_attributes = True
  datasets = read_datasets( datafiles , normalize_attributes, train_data_ratio )
  
  print datasets
  
  ans_matrix = []
  for dataset in datasets:
    k_value_accuracy = []
    for k_value in k_values:
      accuracy_sum = 0 
      for query_idx in dataset.shape[0]:
        query = dataset[query_idx,:][attributes][testing]
        real_class = dataset[query_idx,:][classes][testing]
        predicted_class = k_nn_weighted(query, k_value, dataset)
        if predicted_class == real_class:
          accuracy_sum += 1
      dataset_accuracy = accuracy_sum / dataset
      k_value_accuracy.append(dataset_accuracy)
    ans_matrix.append(k_value_accuracy)
  print ans_matrix
  
def main():
  solve_problem1()  
  
if __name__ == "__main__":
  main()