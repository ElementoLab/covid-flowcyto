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

from src.conf import *


for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

original = pd.read_excel(
    original_dir / ORIGINAL_FILE_NAME,
    na_values=["na", "NT"],
    # converters={"percents": lambda value: "{}%".format(value * 100)},
)

# extract only metadata
meta = original.iloc[:, :N_CATEGORICAL_COLUMNS]


meta.columns = meta.columns.str.strip()

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

meta["age"] = meta["age"].astype(float)
meta["sex"] = (
    meta["sex"].str.replace("m", "Male", case=False).str.replace("f", "Female", case=False)
)


# reverse the name of death/live
meta["death"] = meta["alive"]

# hospitalization
meta["hospitalization"] = meta["hospitalized"].replace("no", "False").replace("yes", "True")

# intubation
meta["intubation"] = (
    meta["intubated"].replace("not intubated", "False").replace("intubated", "True")
)


# clean timepoint
meta["time_symptoms"] = (
    meta["time days from symptoms start"]
    .replace("unk", np.nan)
    .replace("neg", np.nan)
    .astype(float)
)

# treatment
meta["tocilizumab"] = meta["tocilizumab"].replace("no", "False").replace("yes", "True")

# # pre-post treatment samples
t = meta["tocilizumab"] == "True"
meta.loc[t, "tocilizumab_pretreatment"] = (
    meta.loc[t, "datesamples"] < meta.loc[t, "date_tocilizumab"]
).astype(str)
meta.loc[t, "tocilizumab_postreatment"] = (
    meta.loc[t, "datesamples"] > meta.loc[t, "date_tocilizumab"]
).astype(str)

# # extract obesity
# meta.loc[meta["other"].str.contains("nonobese"), "obesity"] = "nonobese"
# meta.loc[meta["other"].str.contains("overweight"), "obesity"] = "overweight"
# meta.loc[
#     (~meta["other"].str.contains("nonobese")) & meta["other"].str.contains("obese"), "obesity"
# ] = "obese"
meta["bmi"] = meta["bmi"].astype(float)


meta["heme"] = meta["heme"].replace("no", "False").replace("yes", "True")
meta["bmt"] = meta["bmt"].replace("no", "False").replace("yes", "True")
meta["pcr"] = meta["pcr"].replace("neg", "False").replace("pos", "True")

# extract commorbidities

# # leukemia/lymphoma
lymp = ["CLL", "AML", "DLBCL", "MM", "ALL"]
meta["leukemia-lymphoma"] = "False"
meta.loc[
    meta["other"].str.contains("|".join(lymp), case=False).fillna(False), "leukemia-lymphoma"
] = "True"


# # Hypertension
# meta["hypertension"] = "False"
# meta.loc[meta["other"].str.contains("HTN", case=False), "hypertension"] = "True"
meta["hypertension"] = meta["HTN"].replace("yes", "True")
# for patients in the 'mild' and 'severe' groups, assume NaN means False
meta.loc[
    pd.isnull(meta["hypertension"]) & meta["severity_group"].isin(["severe", "mild"]),
    "hypertension",
] = "False"


# # DM - TODO: find out what it is
meta["DM"] = meta["DM"].replace("no", "False").replace("yes", "True")

# # Hyperlypidemia
meta["hyperlypidemia"] = "False"
meta.loc[meta["other"].str.contains("HL", case=False).fillna(False), "hyperlypidemia"] = "True"


# # Sleep apnea
meta["sleep_apnea"] = "False"
meta.loc[meta["other"].str.contains("OSA", case=False).fillna(False), "sleep_apnea"] = "True"


# Cleanup strings
for col in meta.loc[:, meta.dtypes == "object"]:
    if all(meta[col].value_counts().index.isin(["False", "True"])):
        continue
    meta[col] = meta[col].str.strip()

# Normal donors vs patients
meta["patient"] = (
    (meta["severity_group"] != "negative").replace(False, "Control").replace(True, "Patient")
)


# COVID19 vs rest
meta["COVID19"] = (~meta["severity_group"].isin(["negative", "non-covid"])).astype(str)


# make ordered Categorical
categories = {
    "patient": ["Control", "Patient"],
    "COVID19": ["False", "True"],
    "severity_group": ["negative", "non-covid", "mild", "severe", "convalescent"],
    "hospitalization": ["False", "True"],
    "intubation": ["False", "True"],
    "death": ["alive", "mild", "dead"],
    "heme": ["False", "True"],
    "bmt": ["False", "True"],
    "obesity": ["nonobese", "overweight", "obese"],
    # TODO: pyarrow/parquet serialization currently does not support categorical bool
    "leukemia-lymphoma": ["False", "True"],
    "hypertension": ["False", "True"],
    "hyperlypidemia": ["False", "True"],
    "sleep_apnea": ["False", "True"],
    "DM": ["False", "True"],
    "tocilizumab": ["False", "True"],
    "tocilizumab_pretreatment": ["False", "True"],
    "tocilizumab_postreatment": ["False", "True"],
    "pcr": ["False", "True"],
}

for col in categories:
    meta[col] = pd.Categorical(meta[col], categories=categories[col], ordered=True)

# reorder columns
last_cols = ["other", "flow_comment"]
order = meta.columns[~meta.columns.isin(last_cols)].tolist() + last_cols
meta = meta[order]
meta.index.name = "sample_name"

to_drop = ["time days from symptoms start", "HTN", "alive", "hospitalized", "intubated"]

meta = meta.drop(to_drop, axis=1)
meta.to_csv(metadata_dir / "annotation.csv")
meta.to_parquet(metadata_dir / "annotation.pq")
# meta.to_hdf(metadata_dir / "annotation.h5", "metadata", format="table")


# extract FACS values
matrix = original.iloc[:, N_CATEGORICAL_COLUMNS:]
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


# Values are read as fraction from the excel
matrix *= 100


# Fix a few variables which are not percentage but ratios

matrix.loc[:, matrix.max() > 105] /= 100


# Save
# matrix.to_csv(data_dir / "matrix.csv")
matrix.sort_index(0).sort_index(1).to_parquet(data_dir / "matrix.pq")
