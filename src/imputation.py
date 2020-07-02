#!/usr/bin/env python

"""
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from fancyimpute import MatrixFactorization
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS, TSNE
from umap import UMAP

import imc
from imc.graphics import to_color_series

from src.conf import *


meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_file)


# Impute missing values
perc = 100 * (matrix.isnull().sum().sum() / matrix.size)
print(f"% missing values in original matrix: {perc:.3f}")

# # Median
matrix_imp_median = matrix.copy()
medians = matrix.median()
for col in matrix.columns[matrix.isnull().any()]:
    matrix_imp_median.loc[matrix[col].isnull(), col] = medians[col]

# KNN
imputer = KNNImputer(n_neighbors=2, weights="uniform")
matrix_imp_KNN = pd.DataFrame(
    imputer.fit_transform(matrix), index=matrix.index, columns=matrix.columns
)

# MF
matrix_imp_MF = pd.DataFrame(
    MatrixFactorization().fit_transform(matrix), index=matrix.index, columns=matrix.columns
)

matrix_imp_MF = matrix_imp_MF.clip(lower=0)
matrix_imp_MF.to_parquet(matrix_imputed_file)
