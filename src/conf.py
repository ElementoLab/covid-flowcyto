#!/usr/bin/env python

"""
"""

from typing import Union
from pathlib import Path as _Path

import pandas as _pd

Series = Union[_pd.Series]
DataFrame = Union[_pd.DataFrame]

figkws = dict(dpi=300, bbox_inches="tight")

original_dir = _Path("data") / "original"
metadata_dir = _Path("metadata")
data_dir = _Path("data")
results_dir = _Path("results")

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

metadata_file = metadata_dir / "annotation.pq"
matrix_file = data_dir / "matrix.pq"
matrix_imputed_file = data_dir / "matrix_imputed.pq"
matrix_imputed_reduced_file = data_dir / "matrix_imputed_reduced.pq"

# Sample metadata
ORIGINAL_FILE_NAME = "clinical_data.joint.20200710.xlsx"
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
    "hyperlypidemia",
    "sleep_apnea",
    "tocilizumab",
    "tocilizumab_pretreatment",
    "tocilizumab_postreatment",
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
    "time_symptoms",
    "datesamples_continuous",
    "processing_batch_continuous",
]
TECHNICAL = ["processing_batch_categorical", "processing_batch_continuous"]
VARIABLES = CATEGORIES + CONTINUOUS


def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min())


def zscore(x, axis=0):
    return (x - x.mean(axis)) / x.std(axis)
