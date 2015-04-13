'''
Cin UFPE - Aprendizagem de Maquina 2015.1
Lista de exercicios
By: Danilo Neves Ribeiro dnr2
'''

import numpy as np
import pandas as pd

#global settings
np.set_printoptions(threshold=15)
np.set_printoptions(precision=4)

#utils
def normalize(dataset):
  row_sums = dataset.sum(axis=1)
  new_matrix = dataset / row_sums[:, np.newaxis]
  return new_matrix

#algorithms


# ========== SOLUTION PROBLEM 1 ========== #
#using dataset iris and transfusion from UCI, headers were removed

def solve_problem1():
  #set parameters
  normalize_attributes = True
  k_values = [1,2,3,5,7,9,11,13,15]
  train_data_ratio = 0.7;
  
  #read data
  datafiles = ["iris.data.txt", "transfusion.data.txt"]  
  datasets = [ np.array( pd.read_csv(file,sep=',', header=None)) for file in datafiles]
  
  map(np.random.shuffle, datasets)
  
  #split datasets in classes and attributes  
  datasets = [ np.split(dataset,[dataset.shape[1]-1],axis=1 ) for dataset in datasets ]  
  
  #data transformations
  if normalize_attributes :
    datasets = [ [normalize(dataset[0].astype(float)),dataset[1]] for dataset in datasets ];        
  
  #create training and test data
  for dataset in datasets :
    split_list = [np.split(attribute,[train_data_ratio * attribute.shape[0]]) for attribute in attributes]
    split_attributes = np.array( split_list )
    (train_datasets, test_datasets) = (split_attributes[:,0], split_attributes[:,1] )

  
  
def main():
  solve_problem1()
  
  
if __name__ == "__main__":
  main()