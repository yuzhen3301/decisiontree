# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:52:44 2019

@author: yuzhen
"""
import pandas as pd

empty_dict = {'age':'', 'workclass':'', 'fnlwgt':'', 'education':'', 'education-num':'', 'marital-status':'', 'occupation':'', 'relationship':'', 'race':'', \
                                       'sex':'', 'capital-gain':'', 'capital-loss':'', 'hours-per-week':'', 'native-country':'', 'class':''}

dict_columns = empty_dict.keys()

data = pd.read_csv('adult.data', names=list(empty_dict.keys()))

mydata = data[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']]

for c in mydata.columns:
    mydata[c] = mydata[c].map(str.strip)

index1 = mydata[mydata['class'] == '<=50K'].index
index2 = mydata[mydata['class'] == '>50K'].index

mydata.loc[index1, 'class'] = 0
mydata.loc[index2, 'class'] = 1

mydata_columns = mydata.columns

for col in mydata_columns:
    mydata = mydata.loc[mydata[col] != '?']

mydata = mydata.drop_duplicates(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
#print(mydata)
#print(len(mydata))

mydata.to_csv("adult1.data", header = 0, index = 0)

##%%%
#for ix in mydata.index:
#    print(ix)
#
#print(type(mydata['workclass']))
#
#for c in mydata.columns:
#    for i in mydata.index:
#        mydata[c][i] = mydata[c][i].strip()
#
#index = mydata['workclass'].index
#for i in index:
#    mydata['workclass'][i] = mydata['workclass'][i].strip()
#    print(i, mydata['workclass'][i])