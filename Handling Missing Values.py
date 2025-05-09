# -*- coding: utf-8 -*-
"""
Created on Sat May 10 01:48:01 2025

@author: esatk

AI RESEARCHER MAHMUT ESAT KOLAY
"""

"""
                                     Handling Missing Values

Dealing with missing data is crucial to ensure your model doesn't break or give biased results.
Below are several common techniques:

"""
#Visualize Missing Data
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df1.isnullcbar=False ,cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

#               Deletion Methods 
#Row-wise Deletion
df_cleaned = df.dropna()

#Column-wise Deletion
df_cleaned = df.drop(columns=["column_name"])

#              Imputation Methods
#Mean Imputation (Numerical)
df["income"] = df["income"].fillna(df["income"].mean())

#Median Imputation (Numerical with outliers)
df["age"] = df["age"].fillna(df["age"].median())

#Mode Imputation (Categorical)
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

#              Forward/Backward Fill
#Forward Fill
df.fillna(method='ffill', inplace=True)

#Backward Fill
df.fillna(method='bfill', inplace=True)

#           Advanced Imputation with scikit-learn
#KNN Imputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)
df_numeric = df.select_dtypes(include=["float64", "int64"])
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

#Iterative Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
