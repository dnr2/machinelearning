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
- PROBLEM 3 - 
#UCI https://archive.ics.uci.edu/ml/datasets/Credit+Approval
  => changed column separator AND
  => changed unknown values '?' by -1 to help with numeric columns
#UCI https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations      
'''

import numpy as np
import pandas as pd
import math
import copy

from collections import defaultdict

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

#vdm distance, constants and global variables
VDM_Q = 2
VDM_C = {}
VDM_N = {}

# ============ UTILS ======================== #

def normalize(dataset):
  new_matrix = dataset  
  for column in range(new_matrix.shape[1]):
    first_val = new_matrix[0][column]
    print column, first_val, type(first_val)
    if isinstance(first_val, (int, long, float,np.int64,np.float32)): 
      c_min = min(new_matrix[:,column].astype(float))
      c_max = max(new_matrix[:,column].astype(float))
      # print new_matrix[:,column].astype(float)
      print c_min, c_max
      for row in range(new_matrix.shape[0]):
        nval = float(new_matrix[row][column] - c_min) / float(c_max-c_min)
        if nval == 1.0:
          print "IS ONE ", row, column, nval, new_matrix[row][column]
        new_matrix[row][column] = nval
        
  return new_matrix

#read datasets from different file sources (headers must be removed)
#each dataset is split into attributes and classes
#after that, each type is split into training and and testing
def read_datasets(datafiles,normalize_attributes,class_last_column):
  #read data  
  datasets = [np.array(pd.read_csv(file,sep=',',header=None,converters={})) for file in datafiles]
  map(np.random.shuffle, datasets)  
  
  #split datasets in two types: classes and attributes, index [0,1]  
  datasets = [ np.split(dataset, [dataset.shape[1]-1] if class_last_column[idx] else [1],axis=1 ) 
    for (idx,dataset) in list(enumerate(datasets)) ]    
  
  #swap attributes and classes for class_last_column
  for idx in range(len(datasets)):
    if not class_last_column[idx]:      
      datasets[idx][attributes], datasets[idx][classes] = datasets[idx][classes], datasets[idx][attributes]  
  
  #data transformations
  # if normalize_attributes :
    # datasets = [ [normalize(dataset[attributes].astype(float)),dataset[classes]]  
      # for dataset in datasets ];
  
  print datasets
  
  #splits data set in training and testing data, index [0,1]
  datasets = [ [ np.split(dataset[type], [int(train_data_ratio * dataset[type].shape[0])]) 
    for type in [attributes,classes]]  for dataset in datasets]  
  return datasets

def compute_VDM(dataset):
  global VDM_N
  global VDM_C
  VDM_N = defaultdict(lambda: EPS,{})
  VDM_C = defaultdict(lambda: 0,{}) 
  
  #precompute the values for N and C in VDM distance
  for row in range(dataset[classes][training].shape[0]):
    cur_class = dataset[classes][training][row,0]    
    VDM_C[cur_class] += 1
  for column in range(dataset[attributes][training].shape[1]):
    first_val = dataset[attributes][training][0,column]
    if isinstance(first_val, (int, long, float)):  
      continue
    val_count = defaultdict(lambda: 0.0,{})
    for row in range(dataset[classes][training].shape[0]):  
      cur_val = dataset[attributes][training][row,column]
      cur_class = dataset[classes][training][row,0]      
      val_count[(cur_val,)] += 1.0      
      val_count[(cur_val,cur_class)] += 1.0      
    for key in val_count:  
      cur_val = key[0]      
      if len(key) == 1:
        VDM_N[(column,cur_val)] += val_count[key]
      else:
        cur_class = key[1]
        VDM_N[(column,cur_val,cur_class)] += val_count[key]
                
# ============ DISTANCES ==================== #
  
def euclidian_dist(vec1, vec2):  
  sum = 0  
  for i in range(len(vec1)):
    sum = sum + math.pow(vec1[i] - vec2[i] , 2)
  return math.sqrt(sum)

def vdm_distance(vec1,vec2):
  sum = 0  
  for i in range(len(vec1)):
    a = vec1[i]
    b = vec2[i]
    for c in VDM_C:
      add = abs((VDM_N[(i,a,c)]/VDM_N[(i,a)]) - (VDM_N[(i,b,c)]/VDM_N[(i,b)]))
      add = math.pow(add, VDM_Q)
      sum += add
  return math.sqrt(sum)

#uses -1 or "-1" to represent unknown values
def hvdm_distance(vec1,vec2):
  sum = 0  
  for i in range(len(vec1)):
    a = vec1[i]
    b = vec2[i]    
    if str(a) == '-1' and str(b) == '-1':
      #both unknown
      add = 0 
    elif str(a) == '-1' or str(b) == '-1':
      #only one is unknown
      add = 1
    elif isinstance(a, (int, long, float)) and isinstance(b, (int, long, float)):
      #case numeric attribute, adds EPS to avoid division by zero
      add = abs(a-b) / ((max(a,b) - min(a,b))+EPS)
    else :
      #case categoric attribute
      for c in VDM_C:        
        
        #TODO when VDM_N is EPS
        # if VDM_N[(i,a)] == EPS or VDM_N[(i,b)] == EPS:
          # print i,a,b
          # print VDM_N[(i,a)], VDM_N[(i,b)]
          
        add = abs((VDM_N[(i,a,c)]/VDM_N[(i,a)]) - (VDM_N[(i,b,c)]/VDM_N[(i,b)]))
        add = math.pow(add, VDM_Q)   
    sum += math.pow(add,2)
  return math.sqrt(sum)
  
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
  counting = defaultdict(lambda: 0,{})
  for instance in k_nearest:      
    # counting.setdefault(instance[1], 0)
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

def solve_knn(datasets,dist_func,compute_VDM_globals):
  #run training with different configurations
  for dataset in datasets:    
    # print dataset
    if compute_VDM_globals:
      compute_VDM(dataset)      
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
  # normalize_attributes = True
  class_last_column = [True,True]
  compute_VDM_globals = False
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,normalize_attributes,class_last_column)    
  solve_knn(datasets,euclidian_dist,compute_VDM_globals)

#using datasets Tic-Tac-Toe Endgame and Congressional Voting from UCI
def solve_problem2():
  #set parameters  
  datafiles = ["tic-tac-toe.data.txt", "house-votes-84.data.txt"]    
  # normalize_attributes = False
  class_last_column = [True,False]
  compute_VDM_globals = True
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,normalize_attributes,class_last_column)    
  solve_knn(datasets,vdm_distance,compute_VDM_globals)

#using datasets Acute Inflammations and Credit Approval from UCI
def solve_problem3():
  #set parameters    
  datafiles = ["crx.data.txt","diagnosis.data.txt"]    
  # normalize_attributes = True #TODO should normalize???
  class_last_column = [True,True]
  compute_VDM_globals = False
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,normalize_attributes,class_last_column)    
  solve_knn(datasets,hvdm_distance,compute_VDM_globals)
  
def main():
  solve_problem1()  
  solve_problem2()
  solve_problem3()
  
if __name__ == "__main__":
  main()