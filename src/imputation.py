#!/usr/bin/env python

"""
"""

from sklearn.preprocessing import LabelEncoder
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
matrix_red_var = matrix_imp_MF.copy()
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
        matrix_red_var = matrix_red_var.drop(var, axis=1)
    else:  # this will never happen
        raise ValueError

matrix_red_var.to_parquet(matrix_imputed_reduced_file)


# TODO: reduced redundancy for e.g. 'CD127_dim_CD25_Br/CD4+' and 'CD127_dim_CD25_Br/LY' populations
# Now in case the same exact population exists with two parents
# sep = matrix_red_var.columns.to_series().str.split("/").apply(pd.Series)
# sep.loc[sep.duplicated(subset=0)]


# Reduce metadata per patient by unique or median
def get_median_or_unique(x: pd.DataFrame) -> pd.Series:
    _res = dict()
    for c in x.columns:
        t = x[c].dtype
        v = x[c]
        if t.type in [np.object_, pd.core.dtypes.dtypes.CategoricalDtypeType]:
            v = v.drop_duplicates().squeeze()
            if isinstance(v, pd.Series):
                continue
            _res[c] = v
        elif t.type is np.datetime64:
            continue
        else:
            _res[c] = v.median()
    return pd.Series(_res).rename_axis("column")


red = meta.groupby("patient_code").apply(get_median_or_unique)
meta_red = pd.pivot(red.reset_index(), "patient_code", "column", 0)


## Reapply the category order
for col in meta.columns[meta.dtypes == "category"]:
    x = meta[col]
    meta_red[col] = pd.Categorical(
        meta_red[col],
        categories=x.value_counts().sort_index().index,
        ordered=True,
    )

meta_red.to_csv(metadata_dir / "annotation.reduced_per_patient.csv")
meta_red.to_parquet(metadata_dir / "annotation.reduced_per_patient.pq")

# Reduce data to median per patient/donor
matrix_red_var_red_pat_median = matrix_red_var.groupby(
    meta["patient_code"]
).median()

matrix_red_var_red_pat_median.to_parquet(
    "data/matrix_imputed_reduced.red_pat_median.pq"
)

# Reduce to earliest timepoint per patient
_res = dict()
for pat, idx in matrix_red_var.groupby(meta["patient_code"]).groups.items():
    x = matrix_red_var.loc[idx]
    m = meta.loc[idx]
    if len(idx) == 1:
        _res[pat] = x.squeeze().rename(pat)
    else:
        try:
            _res[pat] = x.loc[m["time_symptoms"].idxmin()].rename(pat)
        except KeyError:  # healthy donors don't have time
            assert m["severity_group"].iloc[0] in ["negative", "non-covid"]
            _res[pat] = x.median().rename(pat)
res = pd.concat(_res).rename_axis(index=["patient", "cell_type"])
res = pd.pivot(res.reset_index(), "patient", "cell_type", 0)
assert (matrix_red_var_red_pat_median.index == res.index).all()
matrix_red_var_red_pat_early = res

matrix_red_var_red_pat_early.to_parquet(
    "data/matrix_imputed_reduced.red_pat_early.pq"
)

