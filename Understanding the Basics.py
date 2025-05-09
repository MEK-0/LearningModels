# -*- coding: utf-8 -*-
"""
Spyder Editor

AI RESEARCHER MAHMUT ESAT KOLAY

10-05-25 01:10
"""

"""                                 (Understanding the Basics)   

Why is Data Preprocessing Essential?
Data preprocessing is a crucial step in any machine learning pipeline. Raw data is often incomplete, inconsistent, and contains errors. Without proper preprocessing:

Machine learning models may not converge,

Results might be biased or inaccurate,

Training time increases significantly.

Garbage in, garbage out — if the input data is bad, the model will perform poorly no matter how advanced the algorithm is.

 Types of data
 1.Numerical Data
 -quantitative values that can ben measured
 2.Categorical Data
 -Labels or categories without numerical meaning
 3.Textual Data
 -Free-form text,used mainly in NLP
 4.Datatime Data
 -Time-Based information -- useful for time series or feature extraction
                     
                        Common Techniques
1.Numerical -> Scaling, Outlier removal , Binning
2.Categorical -> Encoding(Label,One-Hot)
3.Textual -> Cleaning,Tokenizing,Vectorizing
4.Datatime -> Parsing , Feature extraction , lagging                       
 
"""
import pandas as pd

#1.Numerical
df1 = pd.DataFrame({
    'Age':[22,25,30,29],
    'Salary':[50000,60000,65000,58000]
    })

print(df1.dtypes) # Age , Salary ->int64 , dtype -> object

#2.Categorical

df2 = pd.DataFrame({
     'Gender':['Male','Female','Female','Male'],
     'Department':['IT','HR','Finance','IT']
    }) 

print(df2['Gender'].unique()) #Labels or categories without numerical meaning.

#3.Textual Data
df3 = pd.DataFrame({
    'Comment':['Good service','Very bad experince','Average','Excellent']
    })
print(df3['Comment'][0]) #Needs cleaning, tokenizing, stopword removal, etc.

#4. Datatime Data
#Time-based information — useful for time series or feature extraction.,

df4 = pd.DataFrame({
    'Join_Date':pd.to_datetime(['2020-01-01', '2021-05-15', '2022-07-30', '2023-03-12'])
    })
df4['Year'] = df4['Join_Date'].dt.year
df4['Month'] = df4['Join_Date'].dt.month
print(df4)