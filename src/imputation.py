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
print(f"% missing values in original matrix: {perc:.3f}")  # 0.211%

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
    MatrixFactorization().fit_transform(matrix),
    index=matrix.index,
    columns=matrix.columns,
)

matrix_imp_MF = matrix_imp_MF.clip(lower=0)
matrix_imp_MF.to_parquet(matrix_imputed_file)


# Reduce redundancy in variables by getting only the most "differentiated"
# parent for each variable
matrix_reduced = matrix_imp_MF.copy()
for var in matrix_imp_MF.columns:
    try:
        child, parent = var.split("/")
    except ValueError:  # variable has no parent
        continue

    # look for variables named {CHILD}_{PARENT}
    matches = matrix_imp_MF.columns.str.startswith(f"{child}_{parent}/")
    if not matches.any():
        continue
    if matches.sum() == 1:
        diff = matrix_imp_MF.columns[matches][0]
        matrix_reduced = matrix_reduced.drop(var, axis=1)
    else:  # this will never happen
        raise ValueError

# TODO: reduced redundancy for e.g. 'CD127_dim_CD25_Br/CD4+' and 'CD127_dim_CD25_Br/LY' populations
# Now in case the same exact population exists with two parents
# sep = matrix_reduced.columns.to_series().str.split("/").apply(pd.Series)
# sep.loc[sep.duplicated(subset=0)]

matrix_reduced.to_parquet(matrix_imputed_reduced_file)
