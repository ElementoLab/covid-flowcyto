#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import imc


def minmax_scale(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - x.min()) / (x.max() - x.min())


figkws = dict(dpi=300, bbox_inches="tight")

original_dir = Path("data") / "original"
metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")
output_dir = results_dir / "clinical"
output_dir.mkdir(exist_ok=True, parents=True)

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

metadata_file = metadata_dir / "annotation.pq"
matrix_file = data_dir / "matrix.pq"
matrix_imputed_file = data_dir / "matrix_imputed.pq"


categories = ["patient", "severity_group", "intubated", "death", "heme", "bmt", "obesity"]
continuous = ["timepoint"]
variables = categories + continuous


meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file)

cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns
parent_population = cols[1].rename("parent_population")


# Variable correlation

to_corr = meta.drop_duplicates(subset="patient_code").copy()

for col in meta.columns[meta.dtypes == "category"]:
    to_corr[col] = meta[col].cat.codes

for col in meta.columns[meta.dtypes.apply(lambda x: x.name).str.contains("datetime")]:
    to_corr[col] = minmax_scale(meta[col])

corrs = to_corr.corr(method="spearman").drop("patient", 0).drop("patient", 1)

grid = sns.clustermap(corrs, center=0, cmap="RdBu_r", cbar_kws=dict(label="Spearman correlation"),)
grid.savefig(output_dir / "clinial_parameters.parameter_correlation.svg", **figkws)

# Patient correlation in clinical variables only


grid = sns.clustermap(
    to_corr.query("patient == 1").loc[:, corrs.index].T.corr(method="spearman"),
    center=0,
    cmap="RdBu_r",
    cbar_kws=dict(label="Spearman correlation"),
    xticklabels=True,
    yticklabels=True,
)
grid.savefig(output_dir / "clinial_parameters.sample_correlation.svg", **figkws)

# Add lock file
open(output_dir / "__done__", "w")
