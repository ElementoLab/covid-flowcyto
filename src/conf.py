#!/usr/bin/env python

"""
"""

import json
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from imc.types import Path
import imc  # this import is here to allow automatic colorbars in clustermap


def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min())


def zscore(x, axis=0):
    return (x - x.mean(axis)) / x.std(axis)


def log_pvalues(x, f=0.1):
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = -np.log10(x)
        rmax = ll[ll != np.inf].max()
        return ll.replace(np.inf, rmax + rmax * f)


def text(x, y, s, ax=None, **kws):
    if ax is None:
        ax = plt.gca()
    return [ax.text(x=_x, y=_y, s=_s, **kws) for _x, _y, _s in zip(x, y, s)]


Series = Union[pd.Series]
DataFrame = Union[pd.DataFrame]

figkws = dict(dpi=300, bbox_inches="tight")

original_dir = Path("data") / "original"
metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

metadata_file = metadata_dir / "annotation.pq"
matrix_file = data_dir / "matrix.pq"
matrix_imputed_file = data_dir / "matrix_imputed.pq"
matrix_imputed_reduced_file = data_dir / "matrix_imputed_reduced.pq"

# Sample metadata
ORIGINAL_FILE_NAME = "clinical_data.joint.20200710.csv"
N_CATEGORICAL_COLUMNS = 25

# # variables
CATEGORIES = [
    "sex",
    "patient",
    "COVID19",
    "severity_group",
    "hospitalization",
    "intubation",
    "death",
    "heme",
    "bone_marrow_transplant",
    "obesity",
    "leukemia_lymphoma",
    "diabetes",
    "hypertension",
    # "hyperlypidemia",
    "sleep_apnea",
    "tocilizumab",
    # "tocilizumab_pretreatment",
    # "tocilizumab_postreatment",
    "pcr",
    "processing_batch_categorical",
]
CATEGORIES_T1 = [
    "sex",
    "patient",
    "COVID19",
    "severity_group",
    "hospitalization",
    "intubation",
    "death",
    "diabetes",
    "obesity",
    "hypertension",
    "tocilizumab",
]
CONTINUOUS = [
    "age",
    "bmi",
    "time_symptoms",
    "datesamples_continuous",
    "processing_batch_continuous",
]
TECHNICAL = ["processing_batch_categorical", "processing_batch_continuous"]
VARIABLES = CATEGORIES + CONTINUOUS


try:
    meta = pd.read_parquet(metadata_file)
    matrix = pd.read_parquet(matrix_imputed_file)
    matrix_reduced = pd.read_parquet(matrix_imputed_reduced_file)
    categories = CATEGORIES
    continuous = CONTINUOUS
    sample_variables = meta[categories + continuous]

    cols = matrix.columns.str.extract("(.*)/(.*)")
    cols.index = matrix.columns
    parent_population = cols[1].rename("parent_population")

    panel_variables = json.load(open(metadata_dir / "panel_variables.json"))
    panel_variables = {x: k for k, v in panel_variables.items() for x in v}
    panel = {col: panel_variables[col] for col in matrix.columns}

    variable_classes = (
        parent_population.to_frame()
        .join(pd.Series(panel, name="panel"))
        .join(matrix.mean().rename("Mean"))
        .join(
            matrix.loc[meta["patient"] == "Control"]
            .mean()
            .rename("Mean control")
        )
        .join(
            matrix.loc[meta["patient"] == "Patient"]
            .mean()
            .rename("Mean patient")
        )
    )
except:
    pass
