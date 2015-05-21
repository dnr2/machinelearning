'''
Cin UFPE - Aprendizagem de Maquina 2015.1
Lista de Exercicios #3
By: Danilo Neves Ribeiro dnr2@cin.ufpe.br

UCI Datasets:
- PROBLEM 1 - 
#UCI https://archive.ics.uci.edu/ml/datasets/Iris

followed steps from:
http://www.nlpca.org/pca-principal-component-analysis-matlab.html
'''

import numpy as np
import pandas as pd
import math
import time
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.core import isfinite

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
np.random.seed(23423478)

#normalized euclidian distance global variables
min_val_col = []
max_val_col = []

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
 
#computer rage for euclidean distance normalization
def compute_range_col(dataset):
  new_matrix = dataset  
  global min_val_col
  global max_val_col
  min_val_col = []
  max_val_col = []
  for column in range(new_matrix.shape[1]):
    first_val = new_matrix[0][column]        
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

def pca(data, num_pc):
    
  #normalize data with mean
  mean_vec = [np.mean( data, axis = 0 )]
  repeated_mean_vec = np.repeat(mean_vec, data.shape[0], axis = 0)
  data = np.subtract( data , repeated_mean_vec )
  
  #compute covariance matrix and eigenvectors & eigenvalues
  cov_mat = np.cov( data, rowvar = 0 )
  eigenValues, eigenVectors = np.linalg.eig(cov_mat)
  
  #get num_pc best
  idx = list(reversed(eigenValues.argsort())) # reverse sorting  
  idx = idx[0:num_pc] # selects only num_pc vectors
    
  eigenValues = eigenValues[idx]
  eigenVectors = eigenVectors[:,idx]
  
  data = np.array(  np.matrix(data) * np.matrix(eigenVectors) )
  
  return ( eigenVectors, data )

def lda( data, class_data, num_pc):
    
  class_set = set(class_data)  
  Mall = np.mean( data, axis = 0 )
  
  Sw = None
  Sb = None
  for l in class_set:
    
    print l
    
    indexes = [ idx for idx,ele in enumerate(class_data.tolist()) if ele == l]    
    nl = len( indexes )
    Lsamples = data[indexes,:]
    Ml = np.mean( Lsamples, axis = 0 )    
    diff = np.matrix( Ml - Mall )
    
    if Sb is None :
      Sb = nl * np.transpose(diff) * diff
    else :
      Sb = Sb + (nl * np.transpose(diff) * diff)
    
    for i in range(0,nl):
      diff = np.matrix(Lsamples[i,:] - Ml)
      if Sw is None :
        Sw = np.transpose(diff) * diff
      else :
        Sw = Sw + (np.transpose(diff) * diff)
    print Sb.shape
    print Sw.shape
    
  if np.linalg.det(Sw) == 0:
    #TODO
    print "error!! determinant is zero!"
  
  SwSb = np.asarray(np.linalg.inv(Sw) * Sb)
  print SwSb
  for a in SwSb :
    for b in a:
      print type(b)
      print isfinite(b)
    print isfinite(a.tolist)
  # eigenValues, eigenVectors = np.linalg.eig( SwSb )
  
  #get num_pc best
  idx = list(reversed(eigenValues.argsort())) # reverse sorting  
  idx = idx[0:num_pc] # selects only num_pc vectors
    
  eigenValues = eigenValues[idx]
  eigenVectors = eigenVectors[:,idx]
  
  data = np.array(  np.matrix(data) * np.matrix(eigenVectors) )
  
  return ( eigenVectors, data )
  
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
          line_to_plot.append(dataset_accuracy)
          
        #Not prototyped case
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
      
      #set labels and title and settings, save chart afterwards
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
  
  toy_dataset = datasets[0][attributes][training]
  toy_classes = datasets[0][classes][training][:,0]
  
  # toy_dataset = np.array( [
  # [151,149],[-38,-23],[85,73],[-101,-115]
  # ,[130,137],[-47,-34],[64,72],[-111,-108]
  # ,[139,127],[-39,-49],[67,60],[-115,-128]
  # ,[118,128],[-53,-50],[50,41],[-130,-124]
  # ,[117,102],[-67,-59],[32,41],[-140,-134]
  # ,[104,109],[-68,-77],[35,24],[-142,-152]
  # ,[91,100],[-87,-93],[29,15],[-155,-163]
  # ,[95,182],[-103,-92],[9,15],[-162,-150]
  # ,[-10,-20],[-178,-168],[-23,-16]]
  # )
  
  # eigenVectors, new_toy_dataset = pca( toy_dataset, 3 )
  eigenVectors, new_toy_dataset = lda(toy_dataset, toy_classes, 1)
  
  print eigenVectors
  print new_toy_dataset
  
  # solve_knn(datasets,euclidian_dist_norm)
  
def main():
  # test_plot_settings()
  solve_problem1()    
  
if __name__ == "__main__":
  main()