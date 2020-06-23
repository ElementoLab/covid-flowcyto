#!/usr/bin/env python

"""
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import imc


original_dir = Path("data") / "original"
metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

original = pd.read_excel(original_dir / "COVID-19 Results - Andres.xlsx", na_values="NT")

# extract only metadata
meta = original.iloc[:, :13]


# derive patient/sample codes
meta["patient_code"] = "P" + meta["patient_code"].astype(str).str.zfill(3)
meta["accession"] = "P" + meta["accession"].str.extract(r"P(\d+)-(\d+)").apply(
    lambda x: x.str.zfill(3)
).apply("-".join, axis=1)


for _, idx in meta.groupby(["patient_code", "accession"]).groups.items():
    for i, (ri, row) in enumerate(meta.loc[idx].iterrows(), 1):
        meta.loc[ri, "replicate"] = f"R{str(i).zfill(2)}"


# make sample_name
meta["sample_id"] = "S" + meta["accession"].str.extract(r"P\d+-(\d+)")[0]
meta.index = (
    (meta["patient_code"] + "-" + meta["sample_id"]).rename("sample_name") + "-" + meta["replicate"]
)


# reverse the name of death/live
meta["death"] = meta["alive"]


# clean timepoint
meta["timepoint"] = meta["timepoint"].replace("unk", np.nan).replace("neg", np.nan).astype(float)

# extract obesity
meta.loc[meta["note"].str.contains("nonobese"), "obesity"] = "nonobese"
meta.loc[meta["note"].str.contains("overweight"), "obesity"] = "overweight"
meta.loc[
    (~meta["note"].str.contains("nonobese")) & meta["note"].str.contains("obese"), "obesity"
] = "obese"


# extract commorbidities

# # leukemia/lymphoma
lymp = ["CLL", "AML", "DLBCL", "MM", "ALL"]
meta.loc[meta["note"].str.contains("|".join(lymp)), "comorbidity"] = "leukemia/lymphoma"

# # Hypertension
meta.loc[meta["note"].str.contains("HTL"), "hypertension"] = "hypertension"

# # Hyperlypidemia
meta.loc[meta["note"].str.contains("HL"), "hyperlypidemia"] = "hyperlypidemia"


# # Sleep apnea
meta.loc[meta["note"].str.contains("OSA"), "sleep_apnea"] = "sleep_apnea"


# Cleanup strings
for col in meta.loc[:, meta.dtypes == "object"]:
    meta[col] = meta[col].str.strip()

# Normal donors vs patients
meta["patient"] = "Patient"
meta.loc[meta["note"] == "QC", "patient"] = "Control"

# make ordered Categorical
categories = {
    "severity_group": ["non-covid", "negative", "mild", "severe", "convalescent"],
    "intubated": ["not intubated", "intubated"],
    "death": ["alive", "mild", "dead"],
    "heme": ["no", "yes"],
    "bmt": ["no", "yes"],
    "obesity": ["nonobese", "overweight", "obese"],
    "patient": ["Control", "Patient"],
}

for col in categories:
    meta[col] = pd.Categorical(meta[col], categories=categories[col], ordered=True)

# reorder columns
last_cols = ["note", "flow_comment"]
order = meta.columns[~meta.columns.isin(last_cols)].tolist() + last_cols
meta = meta[order]
meta.index.name = "sample_name"
meta.to_csv(metadata_dir / "annotation.csv")
meta.to_parquet(metadata_dir / "annotation.pq")
# meta.to_hdf(metadata_dir / "annotation.h5", "metadata", format="table")


# extract FACS values
matrix = original.iloc[:, 13:]
matrix.index = meta.index
matrix.columns.name = "cell_type"


# Cleanup redundant columns
for c1 in matrix.columns:
    for c2 in matrix.columns:
        try:  #  in case the column was already removed
            same = (matrix[c1] == matrix[c2]).all()
        except KeyError:
            continue
        if same and c1 != c2 and c1 in c2:
            print(f"Columns '{c1}' and '{c2}' are the same.")
            matrix = matrix.drop(c2, axis=1)


# Save
# matrix.to_csv(data_dir / "matrix.csv")
matrix.sort_index(0).sort_index(1).to_parquet(data_dir / "matrix.pq")
