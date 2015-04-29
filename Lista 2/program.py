'''
Cin UFPE - Aprendizagem de Maquina 2015.1
Lista de Exercicios #2
By: Danilo Neves Ribeiro dnr2@cin.ufpe.br

UCI Datasets:
- PROBLEM 1 - 
#UCI https://archive.ics.uci.edu/ml/datasets/Iris
#UCI https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center

Used website references:
http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/knn.pdf
http://www.eicstes.org/EICSTES_PDF/PAPERS/The%20Self-Organizing%20Map%20%28Kohonen%29.pdf
http://www.mathworks.com/help/nnet/ug/learning-vector-quantization-lvq-neural-networks-1.html
http://users.ics.aalto.fi/mikkok/thesis/book/node20.html
'''

import numpy as np
import pandas as pd
import math
import time
import copy
import matplotlib.pyplot as plt

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
# np.random.seed(94319287)

#normalized euclidian distance global variables
min_val_col = []
max_val_col = []

#LVQ constants
LVQ_NUM_ITERATIONS = 100
learing_rate = 0.02
w_value = 0.2
epsilon_value = 0.1
ratio_num_prototypes_range = [0.10,0.15,0.20,0.25,0.30,0.35]
algorithm_str_range = ["K-NN","LVQ1","LVQ2.1","LVQ3"]  

#chart and plot settings

chart_titles = ["Iris Dataset LVQ", "Transfusion Dataset LVQ" ]

plot_settings = { 
  ("K-NN",1) : ['o','--','k','Not Prototyped, k = 1'],
  ("K-NN",3) : ['o','--','k','Not Prototyped, k = 3'],
  
  ("LVQ1",1) : ['^','-','r','LVQ1, k = 1'],
  ("LVQ1",3) : ['^','-','r','LVQ1, k = 3'],
  
  ("LVQ2.1",1) : ['h','-','b','LVQ2.1, k = 1'],
  ("LVQ2.1",3) : ['h','-','b','LVQ2.1, k = 3'],
  
  ("LVQ3",1) : ['D','-','g','LVQ3, k = 1'],
  ("LVQ3",3) : ['D','-','g','LVQ3, k = 3'],
}

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
  prototypes = [ dataset[type][training][indexes,:].copy() for type in [attributes,classes]]
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

  
#returns a new array with prototypes created from the dataset using algorithms
#algorithm_str define the algorithms among: LVQ1, LVQ2.1, LVQ3, NONE
def LVQ_helper(algorithm_str, dataset,ratio_num_prototypes,learing_rate,w_value,epsilon_value): 
  if algorithm_str != "LVQ1" and algorithm_str != "LVQ2.1" and algorithm_str != "LVQ3":
    return [ dataset[type][training].copy() for type in [attributes,classes]]
  elif algorithm_str == "LVQ1":
    prototypes = LVQ_generate_prototypes(dataset,ratio_num_prototypes,learing_rate)      
  elif algorithm_str == "LVQ2.1" or algorithm_str == "LVQ3":
    prototypes = LVQ_helper("LVQ1",dataset,ratio_num_prototypes,learing_rate,w_value,epsilon_value); 
    # print "dataset after LVQ1 = ", dataset    
    # print "prototypes from LVQ1 = ", prototypes
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
      
      elif algorithm_str == "LVQ3":
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
            
  return prototypes
  
  
  
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
 
  k_nn_weighted = False
  
  title_idx = 0
  
  for dataset in datasets:  
    compute_range_col(dataset[attributes][training])
        
    print "#-------------#"        
    
    for k_value in k_values:
      print "k_value =", k_value
        
      for algorithm_str in algorithm_str_range :    
        print "using", algorithm_str
        
        line_to_plot = []
        
        for ratio_num_prototypes in ratio_num_prototypes_range :
          
          if algorithm_str == algorithm_str_range[0] and ratio_num_prototypes > ratio_num_prototypes_range[0]:
            break
            
          prototypes = LVQ_helper(algorithm_str,dataset,ratio_num_prototypes,learing_rate,w_value,epsilon_value);    
          dataset_prototyped = [[None,None],[None,None]]
          
          dataset_prototyped[attributes][training] = prototypes[attributes].copy()
          dataset_prototyped[attributes][testing] = dataset[attributes][testing].copy()
          
          dataset_prototyped[classes][training] = prototypes[classes].copy()
          dataset_prototyped[classes][testing] = dataset[classes][testing].copy()   
          
          accuracy_sum = 0
          for query_idx in range(dataset_prototyped[attributes][testing].shape[0]):
            query = dataset_prototyped[attributes][testing][query_idx,:]        
            real_class = dataset_prototyped[classes][testing][query_idx,0]
            predicted_class = k_nn_predict_class(query, k_value, dataset_prototyped, k_nn_weighted, dist_func)        
            if predicted_class == real_class:
              accuracy_sum += 1
          dataset_accuracy = float(accuracy_sum) / float(dataset_prototyped[attributes][testing].shape[0])
          if dataset_accuracy < 1.0 - dataset_accuracy:
            print "bugou no caso " + k_value + " " + algorithm_str
            print "Acuracia = ", dataset_accuracy
          
          dataset_accuracy = max(dataset_accuracy, 1.0 - dataset_accuracy)
          line_to_plot.append(dataset_accuracy)
          
        #No prototyping case
        while len(line_to_plot) == 1 :
          line_to_plot.append(line_to_plot[0])

        print line_to_plot
        
        #plot line in chart
        x_values = ratio_num_prototypes_range
        y_values = line_to_plot
        if len(y_values) == 2 :
          x_values = [x_values[0], x_values[-1]]

        plt.plot( x_values, y_values,
            marker = plot_settings[(algorithm_str,k_value)][0],
            linestyle = plot_settings[(algorithm_str,k_value)][1],
            color = plot_settings[(algorithm_str,k_value)][2],
            label = plot_settings[(algorithm_str,k_value)][3],
            linewidth = 2.5,
            markersize= 9)
      
      plt.xlabel("number of prototypes (ratio)")
      plt.ylabel("Accuracy")
      plt.title( chart_titles[title_idx], fontdict ={'fontsize' : 20 } )
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)        
      
      plt.grid()
      plt.subplots_adjust(right=0.71,left = 0.08)
      plt.axis( [ratio_num_prototypes_range[0]-0.01, ratio_num_prototypes_range[-1]+0.01, 0.5, 1.1])
      
      fig = plt.gcf()
      fig.set_size_inches(11,5)
      fig.savefig( chart_titles[title_idx] + "_knn_" + str( k_value) + '.png', dpi=200)
      plt.clf() #clear plot
    
    title_idx += 1
      
    
#using dataset iris and transfusion from UCI, headers were removed
def solve_problem1():
  #set parameters      
  datafiles = ["iris.data.txt", "transfusion.data.txt" ]  
  
  class_last_column = [True,True,False]  
  
  #read data and run k_nn algorithm
  datasets = read_datasets(datafiles,class_last_column)
  
  solve_knn(datasets,euclidian_dist_norm)

def test_plot_settings():
  lines_to_plot = [  
    [0.9333333333, 0.9333333333],
    [0.9777777778,	1,	0.9333333333,	0.9777777778,	0.9555555556,	0.9333333333],
    [0.9555555556,	0.9555555556,	0.9111111111,	1,	0.9333333333,	0.9555555556],
    [0.679,	0.9777777778,	1,	0.9777777778,	0.9777777778,	0.9777777778]
  ]
  
  k_value = 1
  
  algorithm_str_idx = 0
  
  for line_to_plot in lines_to_plot:
    
    x_values = ratio_num_prototypes_range
    y_values = line_to_plot
    algorithm_str = algorithm_str_range[algorithm_str_idx]
    algorithm_str_idx += 1
    
    if len(y_values) == 2 :
      x_values = [x_values[0], x_values[-1]]
    
    plt.plot( x_values, y_values,
        marker = plot_settings[(algorithm_str,k_value)][0],
        linestyle = plot_settings[(algorithm_str,k_value)][1],
        color = plot_settings[(algorithm_str,k_value)][2],
        label = plot_settings[(algorithm_str,k_value)][3],
        linewidth = 2.5,
        markersize= 9)
    
  plt.xlabel( "number of prototypes (ratio)" )
  plt.ylabel( "Accuracy" )
  plt.title('Iris Dataset LVQ', fontdict ={'fontsize' : 20 } )
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.7), fancybox=True, shadow=True)        

  plt.grid()
  plt.subplots_adjust(right=0.71,left = 0.08)
  plt.axis( [ratio_num_prototypes_range[0]-0.01, ratio_num_prototypes_range[-1]+0.01, 0.5, 1.1])
  
  fig = plt.gcf()
  fig.set_size_inches(11,5)
  fig.savefig('graph2.png', dpi=200)
  
def main():
  # test_plot_settings()
  solve_problem1()    
  
if __name__ == "__main__":
  main()