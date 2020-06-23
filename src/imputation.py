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


figkws = dict(dpi=300, bbox_inches="tight")

original_dir = Path("data") / "original"
metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

metadata_file = metadata_dir / "annotation.pq"
matrix_file = data_dir / "matrix.pq"
matrix_imputed_file = data_dir / "matrix.pq"

meta = pd.read_parquet(metadata_dir / "annotation.pq")
matrix = pd.read_parquet(data_dir / "matrix.pq")

categories = ["severity_group", "intubated", "death", "heme", "bmt", "obesity", "patient"]
continuous = ["timepoint"]


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
matrix_imp_MF.to_parquet(data_dir / "matrix_imputed.pq")
