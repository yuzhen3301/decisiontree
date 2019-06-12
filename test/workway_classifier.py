# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:29:06 2019

@author: yuzhen
"""

import pandas as pd
import math as mt

dict_attr2vals = {'天气':'', '心情':'', '是否快要迟到':'', 'class':''}

columns = dict_attr2vals.keys()

ds = pd.read_csv('workway.data', names=list(columns))

decision_tree = {}

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
            #decision_tree[ds_attr_optimal][val] = {str(ds_sub['class'].tolist()[0]) : len(ds_sub)} # class : support_num
            decision_tree[ds_attr_optimal][val] = {str(ds_sub['class'].tolist()[0]) : ''} # class : support_num
        elif len(ds_sub_columns) <= 1:
            dict_ds_sub_cls_stat = dict(ds_sub['class'].value_counts().sort_index())
            result_class = max(dict_ds_sub_cls_stat, key = dict_ds_sub_cls_stat.get)
            #decision_tree[ds_attr_optimal][val] = {str(result_class) : dict_ds_sub_cls_stat[result_class]} # class : support_num
            decision_tree[ds_attr_optimal][val] = {str(result_class) : ''} # class : support_num
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
        for val in dt[attr_up]:
            if val == '':
                continue
            else:
                for attr_down in dt[attr_up][val]:
                    dtf.write(indent+"\"" + attr_up + "\" -> \"" + attr_down + "\" [label = \"" + val + "\" ]\n")
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
buildtree(ds, decision_tree)
#prunetree(decision_tree)
##%%%
#
dtf = open("workway_dt.dot", "w")
output_decision_tree(decision_tree, dtf)
dtf.close()