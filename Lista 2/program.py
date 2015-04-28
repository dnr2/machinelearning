'''
Cin UFPE - Aprendizagem de Maquina 2015.1
Lista de Exercicios #2
By: Danilo Neves Ribeiro dnr2@cin.ufpe.br

UCI Datasets:
- PROBLEM 1 - 
#UCI https://archive.ics.uci.edu/ml/datasets/Iris
#UCI https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
'''

import numpy as np
import pandas as pd
import math
import time

from collections import defaultdict

# ============ GLOBAL SETTINGS ============== #

#constants
attributes, classes = (0,1)
training, testing = (0,1)
EPS = 1e-9
k_values = [1,3]
train_data_ratio = 0.7;
number_types = (int, long, float,np.int64,np.float32)

#numpy settings
np.set_printoptions(threshold=15)
np.set_printoptions(precision=4)
np.random.seed(94319287)

#normalized euclidian distance global variables
min_val_col = []
max_val_col = []

#LVQ constants
LVQ_NUM_ITERATIONS = 100

# ============ UTILS ======================== #

#linear decreasing learning rate function
def decreasing_learning_rate(learing_rate,t,total_t):  
  return learing_rate * ((1.0 + EPS) - (float(t)/float(total_t)))
 
#computer rage for euclidian disntance normalization
def compute_range_col(dataset):
  new_matrix = dataset  
  global min_val_col
  global max_val_col
  min_val_col = []
  max_val_col = []
  for column in range(new_matrix.shape[1]):
    first_val = new_matrix[0][column]    
    # print column, type(first_val), first_val
    if isinstance(first_val, number_types): 
      min_val_col.append(min(new_matrix[:,column].astype(float)))
      max_val_col.append(max(new_matrix[:,column].astype(float)))
    else:
      min_val_col.append(0)
      max_val_col.append(0)
  return new_matrix

#read datasets from different file sources (headers must be removed)
#each dataset is split into attributes and classes
#after that, each type is split into training and and testing
def read_datasets(datafiles,class_last_column):
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
  
  # print datasets
  
  #splits data set in training and testing data, index [0,1]
  datasets = [ [ np.split(dataset[type], [int(train_data_ratio * dataset[type].shape[0])]) 
    for type in [attributes,classes]]  for dataset in datasets]  
  return datasets

# ============ DISTANCES ==================== #

def euclidian_dist_norm(vec1, vec2):  
  sum = 0  
  for i in range(len(vec1)):   
    range_col = max_val_col[i] - min_val_col[i]
    sum = sum + math.pow(float(vec1[i] - vec2[i])/float(range_col), 2)
  return math.sqrt(sum)

# ============ ALGORITHMS =================== #

def LVQ_generate_prototypes(dataset,ratio_num_prototypes,learing_rate):
  num_prototypes = int(dataset[attributes][training].shape[0] * ratio_num_prototypes)
  indexes = range(dataset[attributes][training].shape[0])
  np.random.shuffle(indexes)
  indexes = indexes[0:num_prototypes]
  prototypes = [dataset[type][training][indexes,:] for type in [attributes,classes]]
  return prototypes

def get_k_closest_prototypes(k_value,sample_attribute,prototypes):
  closest_prototypes = [];    
  for proto_row in range(prototypes[attributes].shape[0]):     
    proto_att = prototypes[attributes][proto_row]    
    cur_dist = euclidian_dist_norm(proto_att,sample_attribute)
    if len(closest_prototypes) < k_value:
      closest_prototypes.append( (proto_row,cur_dist))
    else:
      sorted( closest_prototypes, key= lambda x : x[1], reverse=True )
      if closest_prototypes[0][1] > cur_dist :
        closest_prototypes[0] = (proto_row,cur_dist) 

  sorted( closest_prototypes, key= lambda x : x[1])        
  return closest_prototypes
  
def LVQ1(dataset,ratio_num_prototypes,learing_rate):

  prototypes = LVQ_generate_prototypes(dataset,ratio_num_prototypes,learing_rate)  

  for iteration in range(LVQ_NUM_ITERATIONS):
    for row in range(dataset[attributes][training].shape[0]):
      sample_attribute = dataset[attributes][training][row]
      sample_class = dataset[classes][training][row]
      
      closest_prototypes = get_k_closest_prototypes(1,sample_attribute,prototypes)
      
      proto_row = closest_prototypes[0][0]
      proto_att = prototypes[attributes][proto_row]
      proto_class = prototypes[classes][proto_row]
      
      cur_iteration_rate = decreasing_learning_rate(learing_rate,iteration,LVQ_NUM_ITERATIONS)
      dir_vec = (sample_attribute - proto_att) * cur_iteration_rate
      
      if sample_class == proto_class:
        prototypes[attributes][proto_row] += dir_vec
      else:
        prototypes[attributes][proto_row] -= dir_vec
    
  dataset[attributes][training] = prototypes[attributes]
  dataset[classes][training] = prototypes[classes]

  return dataset

def LVQ2(dataset,ratio_num_prototypes,learing_rate,w_value):
  
  dataset = LVQ1(dataset,ratio_num_prototypes,learing_rate)
  prototypes = [dataset[type][training][:,:] for type in [attributes,classes]]  
  
  s_value = (1.0-w_value)/(1.0+w_value)
  
  for iteration in range(LVQ_NUM_ITERATIONS):
    for row in range(dataset[attributes][training].shape[0]):
      sample_attribute = dataset[attributes][training][row]
      sample_class = dataset[classes][training][row]
      
      closest_prototypes = get_k_closest_prototypes(2,sample_attribute,prototypes)
      
      proto_row1 = closest_prototypes[0][0]
      dist_proto1 = closest_prototypes[0][1] + EPS
      proto_att1 = prototypes[attributes][proto_row1]
      proto_class1 = prototypes[classes][proto_row1]
      
      proto_row2 = closest_prototypes[1][0]
      dist_proto2 = closest_prototypes[1][1] + EPS
      proto_att2 = prototypes[attributes][proto_row2]
      proto_class2 = prototypes[classes][proto_row2]
      
      cur_iteration_rate = decreasing_learning_rate(learing_rate,iteration,LVQ_NUM_ITERATIONS)
      
      dir_vec1 = (sample_attribute - proto_att1) * cur_iteration_rate
      dir_vec2 = (sample_attribute - proto_att2) * cur_iteration_rate
      
      if min( float(dist_proto1)/dist_proto2 , float(dist_proto2)/dist_proto1 ) > s_value :
        if proto_class1 != proto_class2 : 
          if sample_class == proto_class1:
            prototypes[attributes][proto_row1] += dir_vec1
            prototypes[attributes][proto_row2] -= dir_vec2
          else:
            prototypes[attributes][proto_row1] -= dir_vec1
            prototypes[attributes][proto_row2] += dir_vec2
  
  dataset[attributes][training] = prototypes[attributes]
  dataset[classes][training] = prototypes[classes]
  
  return dataset

  
def LVQ_helper(algorithm_str, dataset,ratio_num_prototypes,learing_rate,w_value,epsilon_value): 
  if algorithm_str == "LVQ1":
    prototypes = LVQ_generate_prototypes(dataset,ratio_num_prototypes,learing_rate)  
  elif algorithm_str == "LVQ2.1" or algorithm_str == "LVQ3":
    dataset = LVQ1(dataset,ratio_num_prototypes,learing_rate)
    prototypes = [dataset[type][training][:,:] for type in [attributes,classes]]  
    s_value = (1.0-w_value)/(1.0+w_value)
    
  for iteration in range(LVQ_NUM_ITERATIONS):
    for row in range(dataset[attributes][training].shape[0]):
      sample_attribute = dataset[attributes][training][row]
      sample_class = dataset[classes][training][row]
      
      closest_prototypes = get_k_closest_prototypes(2,sample_attribute,prototypes)
      
      proto_row1 = closest_prototypes[0][0]
      dist_proto1 = closest_prototypes[0][1] + EPS
      proto_att1 = prototypes[attributes][proto_row1]
      proto_class1 = prototypes[classes][proto_row1]
      
      proto_row2 = closest_prototypes[1][0]
      dist_proto2 = closest_prototypes[1][1] + EPS
      proto_att2 = prototypes[attributes][proto_row2]
      proto_class2 = prototypes[classes][proto_row2]
      
      cur_iteration_rate = decreasing_learning_rate(learing_rate,iteration,LVQ_NUM_ITERATIONS)
      
      dir_vec1 = (sample_attribute - proto_att1) * cur_iteration_rate
      dir_vec2 = (sample_attribute - proto_att2) * cur_iteration_rate
      
      
      if algorithm_str == "LVQ1":
        if sample_class == proto_class1:
          prototypes[attributes][proto_row1] += dir_vec1
        else:
          prototypes[attributes][proto_row1] -= dir_vec1
      
      elif algorithm_str == "LVQ2.1":
      
        if min( float(dist_proto1)/dist_proto2 , float(dist_proto2)/dist_proto1 ) > s_value :
          if proto_class1 != proto_class2 : 
            if sample_class == proto_class1:
              prototypes[attributes][proto_row1] += dir_vec1
              prototypes[attributes][proto_row2] -= dir_vec2
            else:
              prototypes[attributes][proto_row1] -= dir_vec1
              prototypes[attributes][proto_row2] += dir_vec2
      
      elif algorithm_str == "LVQ2.2":
        if min( float(dist_proto1)/dist_proto2 , float(dist_proto2)/dist_proto1 ) > s_value :
          if proto_class1 != proto_class2 : 
            if sample_class == proto_class1:
              prototypes[attributes][proto_row1] += dir_vec1
              prototypes[attributes][proto_row2] -= dir_vec2
            else:
              prototypes[attributes][proto_row1] -= dir_vec1
              prototypes[attributes][proto_row2] += dir_vec2
          elif proto_class1 == proto_class2 and proto_class2 == sample_class:
            prototypes[attributes][proto_row1] += epsilon_value * dir_vec1
            prototypes[attributes][proto_row2] += epsilon_value * dir_vec2
            
  dataset[attributes][training] = prototypes[attributes]
  dataset[classes][training] = prototypes[classes]
  
  return dataset
  
  
  
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

def solve_knn(datasets,dist_func):
  #run training with different configurations
  for dataset in datasets:
    
    ratio_num_prototypes = 0.1  
    learing_rate = 0.02
    
    compute_range_col(dataset[attributes][training])
    
    #=============#
    print "---------"    
    # print "using LVQ1" 
    # dataset = LVQ1(dataset,ratio_num_prototypes,learing_rate);    
    print "using LVQ2" 
    w_value = 0.2
    dataset = LVQ2(dataset,ratio_num_prototypes,learing_rate,w_value);    
    #=============#
    
    start_time = time.time()
    #for k_nn_weighted in [False,True]:   
    for k_nn_weighted in [False]:      
      print "k_nn_weighted", k_nn_weighted
      for k_value in k_values:        
        accuracy_sum = 0
        for query_idx in range(dataset[attributes][testing].shape[0]):
          query = dataset[attributes][testing][query_idx,:]        
          real_class = dataset[classes][testing][query_idx,0]
          predicted_class = k_nn_predict_class(query, k_value, dataset, k_nn_weighted, dist_func)        
          if predicted_class == real_class:
            accuracy_sum += 1
        dataset_accuracy = float(accuracy_sum) / float(dataset[attributes][testing].shape[0])
        print dataset_accuracy,
      print "\n"
    elapsed_time = time.time() - start_time
    # print "elapsed time = ", elapsed_time
    
#using dataset iris and transfusion from UCI, headers were removed
def solve_problem1():
  #set parameters      
  datafiles = ["iris.data.txt", "transfusion.data.txt","wine.data.txt"] 
  class_last_column = [True,True,False]  
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,class_last_column)
  
  solve_knn(datasets,euclidian_dist_norm)

def main():
  solve_problem1()    
  
if __name__ == "__main__":
  main()