# -*- coding: utf-8 -*-
"""
Created on Sat May 10 01:28:55 2025

@author: esatk

AI RESEARCHER MAHMUT ESAT KOLAY
"""

"""
                       Data Loading and Initial Inspection
Loading data and performing an initial inspection are the first steps in any data preprocessing workflow. This helps you understand the structure, quality, and potential problems in the dataset.

"""
import pandas as pd
import numpy as np

#                            Load Dataset

#CSV File
df = pd.read_csv("data.csv")

#Excel File
df1 = pd.read_excel("data.xlsx")

#JSON File
df2 = pd.read_json("data.json")

#From a URL
url = "https://raw.githubusercontent.com/path/to/data.csv"
df3 = pd.read_csv(url)

#                        Preview the Data

#First 5 rows
df.head()

#Dataset Info(Column types,nulls)
df.info()

#Summary Statics(Only Numerical)
df.describe()

#                         Basic Checks
#Shape of the Data
df.shape

#Columns Names
df.columns

#Checking for Missing Values
df.isnull().sum()

#Checking Data Types
df.types

"""
 Identify Potential Issues
Are there any missing values?

Are data types correct? (e.g., date stored as string?)

Are categorical variables encoded properly?

Are there duplicate rows?

"""