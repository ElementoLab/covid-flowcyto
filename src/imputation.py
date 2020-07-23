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


# Convert percentages to cells per microliter
var = "WBC_CBC"
m = meta.dropna(subset=[var])
d = matrix_imp_MF.loc[m.index]
d.sort_index().to_csv("data/matrix_imputed.percentages.csv")
cols = (
    d.columns.to_series()
    .str.split("/")
    .apply(pd.Series)
    .rename(columns={0: "child", 1: "parent"})
)

counts = pd.DataFrame(index=d.index, columns=d.columns)
# extended because the order matters
for col in cols.loc[cols["parent"].str.contains(r"CD45$|CD45_")].index:
    counts.loc[:, col] = m[var] * d[col] * 1e3
for col in cols.loc[cols["parent"] == "LY"].index:
    counts.loc[:, col] = d["LY/All_CD45"] * d[col]
for col in cols.loc[cols["parent"] == "CD3+"].index:
    counts.loc[:, col] = d["CD3+/LY"] * d[col]
for col in cols.loc[cols["parent"] == "CD4+"].index:
    counts.loc[:, col] = d["CD4+/CD3+"] * d[col]
for col in cols.loc[cols["parent"] == "CD8+"].index:
    counts.loc[:, col] = d["CD8+/CD3+"] * d[col]
for col in cols.loc[cols["parent"] == "CD45RA+_CD4+"].index:
    counts.loc[:, col] = d["CD45RA+_CD4+/CD4+"] * d[col]
for col in cols.loc[cols["parent"] == "CD45RO+_CD4+"].index:
    counts.loc[:, col] = d["CD45RO+_CD4+/CD4+"] * d[col]
for col in cols.loc[cols["parent"] == "CD45RA+_CD8+"].index:
    counts.loc[:, col] = d["CD45RA+_CD8+/CD8+"] * d[col]
for col in cols.loc[cols["parent"] == "CD45RO+_CD8+"].index:
    counts.loc[:, col] = d["CD45RO+_CD8+/CD8+"] * d[col]
for col in cols.loc[cols["parent"] == "CD185+"].index:
    counts.loc[:, col] = d["CD185+/CD4+"] * d[col]
for col in cols.loc[cols["parent"] == "All_NK"].index:
    counts.loc[:, col] = d["All_NK/LY"] * d[col]
for col in cols.loc[cols["parent"] == "CD56+_CD16_Br"].index:
    counts.loc[:, col] = d["CD56+_CD16_Br/All_NK"] * d[col]
for col in cols.loc[cols["parent"] == "CD19+_CD20+"].index:
    counts.loc[:, col] = d["CD19+_CD20+/LY"] * d[col]

assert ~counts.isnull().any().any()
counts = counts.astype(int)
counts.sort_index().to_csv("data/matrix_imputed.counts.csv")
counts.to_parquet("data/matrix_imputed.counts.pq")


n_examples = 16
n = int(np.sqrt(n_examples))
inches = 3
fig, axes = plt.subplots(
    n, n, figsize=(n * inches, n * inches), tight_layout=True
)
axes = axes.flatten()
i = 0
for (name, idx) in cols.groupby("child").groups.items():
    if i == n_examples:
        break
    if len(idx) != 2:
        continue
    parents = set(map(lambda x: x[1], idx.str.split("/")))
    if parents == {"CD4+", "CD8+"}:
        continue
    a = counts[idx[0]]  #  + 1
    b = counts[idx[1]]  #  + 1
    vmin = pd.concat([a, b], 1).min(1).min()
    vmax = pd.concat([a, b], 1).max(1).max()
    axes[i].plot((vmin, vmax), (vmin, vmax), linestyle="--", color="gray")
    axes[i].scatter(a, b, s=2, alpha=0.5)
    axes[i].set(xlabel=idx[0], ylabel=idx[1], xscale="symlog", yscale="symlog")
    i += 1
fig.savefig(
    "results/percentages_to_counts.variable_comparison.example.svg", **figkws
)
