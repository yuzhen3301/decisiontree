# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:29:06 2019

@author: yuzhen
"""

import numpy as np
import pandas as pd
import math as mt

nprand = np.random
training_ratio = 0.75

inputfile = 'adult.data'
outputfile = 'dt.dot'

dict_attr2vals = {'workclass':'', 'education':'', 'marital-status':'', 'occupation':'', 'relationship':'', 'race':'', \
                                       'sex':'', 'native-country':'', 'class':''}

dict_columns = dict_attr2vals.keys()

ds = pd.read_csv(inputfile, names=list(dict_columns))

ds = ds.sort_values(by="class" , ascending=True)

decision_tree = {}

#%%%
def num_of_samples_in_each_class(ds):
    return ds['class'].value_counts().sort_index()

#%%%
#split the dataset into a training set and a validation set
def split_dataset(ds, offset, sample_num, training_ratio):
    training_set = pd.DataFrame([], columns=list(dict_columns))
    validation_set = pd.DataFrame([], columns=list(dict_columns))
    training_set_length = int(sample_num * training_ratio)
    training_sample_indices = []
    while len(training_sample_indices) < training_set_length:
        training_sample_index = nprand.randint(0, sample_num)
        if training_sample_index in training_sample_indices:
            continue
        
        training_sample_indices.append(training_sample_index)
    validation_sample_indices = list(filter(lambda x: x not in training_sample_indices, list(range(sample_num))))
    training_sample_indices.sort()
    validation_sample_indices.sort()

    for i in training_sample_indices:
        training_set = training_set.append(ds.iloc[offset+i])
    
    for i in validation_sample_indices:
        validation_set = validation_set.append(ds.iloc[offset+i])
    
    return training_set, validation_set

#%%%
def calc_entropy_for_dataset(ds):
    ds_size = len(ds)
    ds_cls_stat = ds['class'].value_counts().sort_index()
    ds_entropy = 0.0
    for i in ds_cls_stat.index:
        ds_cls_prop = ds_cls_stat[i]/ds_size
        ds_entropy += ds_cls_prop * (-1.0 * mt.log2(ds_cls_prop))
    
    return ds_entropy

#%%%
def get_optimal_attr(ds):
    ds_columns = ds.columns
    ds_size = len(ds)
    
    dict_attr2vals = dict.fromkeys(tuple(ds_columns), '')
    dict_attr2entropy = dict.fromkeys(tuple(ds_columns), '')
    
    ds_entropy = calc_entropy_for_dataset(ds)
    
    for attr in ds_columns: #drop duplicates for values of a specified column
        dict_attr2vals[attr] = ds[attr].drop_duplicates().tolist()
    
    for attr in ds_columns:
        if attr == 'class':
            continue
        
        ds_attr_entropy = 0.0
        for val in dict_attr2vals[attr]:
            ds_sub = ds[ds[attr] == val].drop(attr, 1)
            
            #H(Di)
            ds_sub_entropy = calc_entropy_for_dataset(ds_sub)
            
            #P(Di)
            ds_sub_size = len(ds_sub)
            ds_sub_prop = ds_sub_size/ds_size
            
            #H(D, attr) = sigma(H(Di)*P(Di))
            ds_attr_entropy += ds_sub_entropy * ds_sub_prop
        
        dict_attr2entropy[attr] = ds_attr_entropy
    
    dict_attr2entropy['class'] = 1.0 #'class' is not a real class of the original dataset
    
    #I(attr) = H(D) - H(D, attr)
    dict_attr2gain = {k : ds_entropy-dict_attr2entropy[k] for k in dict_attr2entropy}
    
    ds_attr_optimal = max(dict_attr2gain, key = dict_attr2gain.get)
    
    return ds_attr_optimal

#%%%
def buildtree(ds, decision_tree):
    ds_attr_optimal = get_optimal_attr(ds)
    decision_tree[ds_attr_optimal] = {}
    
    optimal_attr_vals = ds[ds_attr_optimal].drop_duplicates().tolist()
    
    for val in optimal_attr_vals:
        ds_sub = ds[ds[ds_attr_optimal] == val].drop(ds_attr_optimal, 1)
        ds_sub_columns = ds_sub.columns
        ds_sub_entropy = calc_entropy_for_dataset(ds_sub)
        
        if ds_sub_entropy == 0.0:
            decision_tree[ds_attr_optimal][val] = {str(ds_sub['class'].tolist()[0]) : len(ds_sub)} # class : support_num
        elif len(ds_sub_columns) <= 1:
            dict_ds_sub_cls_stat = dict(ds_sub['class'].value_counts().sort_index())
            result_class = max(dict_ds_sub_cls_stat, key = dict_ds_sub_cls_stat.get)
            decision_tree[ds_attr_optimal][val] = {str(result_class) : dict_ds_sub_cls_stat[result_class]} # class : support_num
        else: #len(ds_sub_columns) > 1
            decision_tree[ds_attr_optimal][val] = {}
            buildtree(ds_sub, decision_tree[ds_attr_optimal][val])

#%%%
def output_decision_tree(dt, dtf):
    dtf.write("digraph dt {\n")
    dtf.write("    node [shape = \"box\", style = \"filled\", color = \"red\"]\n")
    indent = "    "
    dump_decision_tree(dt, dtf, indent)
    dtf.write("}\n")

def dump_decision_tree(dt, dtf, indent):
    for attr_up in dt:
        if not isinstance(dt[attr_up], dict):
            continue
        else:
            for val in dt[attr_up]:
                for attr_down in dt[attr_up][val]:
                    dtf.write(indent+"\"" + attr_up + "\" -> \"" + attr_down + "\" [label = \"" + val + "\"]\n")
                indent_deeper = indent + "    "
                dump_decision_tree(dt[attr_up][val], dtf, indent_deeper)

##%%%
#def prunetree(dt):
#    for attr_up in dt:
#        for val in dt[attr_up]:
#            if isinstance(val, dict) == False:
#                continue
#            else:
#                prunetree(dt[attr_up][val])
#                for attr_down in dt[attr_up][val]:
#        
                
#%%%
def get_class2number_from_tree(dt, map_class2number):
    for attr in dt:
        if not isinstance(dt[attr], dict):
            if not (attr in map_class2number.keys()):
                map_class2number[attr] = int(dt[attr])
            else:
                map_class2number[attr] += int(dt[attr])
        else:
            for val in dt[attr]:
                get_class2number_from_tree(dt[attr][val], map_class2number)

#%%%
def get_dominant_class_from_tree(dt):
    dict_class2number = {}
    get_class2number_from_tree(dt, dict_class2number)
    dominant_class = max(dict_class2number, key = dict_class2number.get)
    return dominant_class

#%%%
def get_sample_class_from_tree(x_sample, dt):
    x_sample_class = ''
    for attr in dt:
        if attr == '0' or attr == '1':
            x_sample_class = attr
            return x_sample_class
        else:
            x_sample_value_found = False
            for val in dt[attr]:
                if x_sample[attr] == val:
                    x_sample_value_found = True
                    x_sample_class = get_sample_class_from_tree(x_sample, dt[attr][val])
                    break
                
            if not x_sample_value_found:
                x_sample_class = get_dominant_class_from_tree(dt[attr][val])
                break
    
    return x_sample_class
    
#%%%
def evaluate_decision_tree_on_validation_set(ds, dt):
    correction_number = 0
    correction_rate = 0.0
    for ix in range(0, ds.shape[0], 1):
        x_sample = ds.iloc[ix]
        x_sample_class = get_sample_class_from_tree(x_sample, dt)
            
        #print("x_sample_class is " + str(x_sample_class) + ", type(x_sample_class) is " + str(type(x_sample_class)))
        #print("x_sample['class'] is " + str(x_sample['class']) + ", type(x_sample['class']) is " + str(type(x_sample['class'])))
        if x_sample_class == str(x_sample['class']): #type(x_sample_class) is str, while type(x_sample['class']) is int
            correction_number += 1
        
    print(correction_number)
    correction_rate = float(correction_number)/ds.shape[0]
    return correction_rate
                        
        
#%%%
#generate the training set and the valdiation set
series_class_to_sample_num = num_of_samples_in_each_class(ds)
print(series_class_to_sample_num)
class1_training_set, class1_validation_set = split_dataset(ds, 0, series_class_to_sample_num.values[0], training_ratio)
class2_training_set, class2_validation_set = split_dataset(ds, series_class_to_sample_num.values[0], series_class_to_sample_num.values[1], training_ratio)

ds_training_set = class1_training_set.append(class2_training_set)
#print(ds_training_set)

ds_validation_set = class1_validation_set.append(class2_validation_set)
#print(ds_validation_set)


buildtree(ds_training_set, decision_tree)
correction_rate = evaluate_decision_tree_on_validation_set(ds_validation_set, decision_tree)
print(correction_rate)

#prunetree(decision_tree)

dtf = open(outputfile, "w")
output_decision_tree(decision_tree, dtf)
dtf.close()