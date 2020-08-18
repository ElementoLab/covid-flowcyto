#!/usr/bin/env python

"""
Generation of supplementary tables
"""

import json
from copy import deepcopy


import pandas as pd  # type: ignore[import]
import numpy as np  # type: ignore[import]

from src.conf import Path, meta, matrix, panels as _panels


def save_excel(df: pd.DataFrame, output_file: Path, sheet_name: str) -> None:
    """
    Write excel file with adjusted column width, frozen rows/columns, and Arial font.
    """
    writer = pd.ExcelWriter(output_file, engine="xlsxwriter")
    df = df.reset_index()
    df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 1))
    worksheet = writer.sheets[sheet_name]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = (
            max(
                (
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name)),  # len of column name/header
                )
            )
            + 3
        )  # adding a little extra space
        fmt = writer.book.add_format({"font_name": "Arial"})
        worksheet.set_column(idx, idx, max_len, fmt)  # set column width
    writer.save()


output_dir = Path("supplement")
output_dir.mkdir()

keep = [
    "patient_code",
    "sample_id",
    "replicate",
    "accession",
    "sex",
    "race",
    "age",
    "severity_group",
    "time_symptoms",
    "hospitalization",
    "intubation",
    "death",
    "tocilizumab",
    "WBC_CBC",
    "lymph_CBC",
    "neutrophils",
    "obesity",
    "bmi",
    "hypertension",
    "diabetes",
]

meta = meta[keep].sort_values(["severity_group", "patient_code", "replicate"])
save_excel(meta, output_dir / "Supplementary_Table2.xlsx", "Sample metadata")

matrix = matrix.reindex(meta.index)
save_excel(
    matrix, output_dir / "Supplementary_Table5.xlsx", "Immune populations"
)


panels = deepcopy(_panels)
m = max(map(len, panels.values()))
for p in panels:
    while len(panels[p]) < m:
        panels[p] += [np.nan]

populations = pd.DataFrame(panels)
save_excel(
    populations, output_dir / "Supplementary_Table4.xlsx", "Immune populations"
)

_panels = json.load(open("metadata/flow_variables2.json", "r"))
panels = deepcopy(_panels)
m = max(map(len, panels.values()))
for p in panels:
    while len(panels[p]) < m:
        panels[p] += [np.nan]
flow = pd.DataFrame(panels)
flow.columns = (
    flow.columns.to_series()
    .str.split("_")
    .apply(lambda x: x[-1])
    .replace("T3", "Tfol")
)
flow = flow.sort_index(1)
save_excel(flow, output_dir / "Supplementary_Table3.xlsx", "Immune panels")


import pingouin as pg
from scipy.stats import fisher_exact
import scipy

meta = meta.sort_values("time_symptoms").drop_duplicates("patient_code")

sevs = meta["severity_group"].cat.categories.tolist()
res = pd.DataFrame(columns=sevs + ["stat", "p-value"])

conts = meta.columns[meta.dtypes == "float64"]
conts = [c for c in conts if c in keep]
for con in conts:
    f = np.mean if con != "age" else np.median
    meta2 = meta.dropna(subset=[con]).copy()
    meta2["severity_group"] = meta2[
        "severity_group"
    ].cat.remove_unused_categories()
    base = meta2["severity_group"].cat.categories[0]
    stats = pg.pairwise_ttests(
        meta2, dv=con, between="severity_group", parametric=False,
    )
    stats = stats.query(f"A == '{base}'")
    est = meta2.groupby("severity_group")[con].apply(f)
    iqr = (
        meta2.dropna(subset=[con])
        .groupby("severity_group")[con]
        .apply(scipy.stats.iqr)
    )
    for sev in meta2["severity_group"].cat.categories:
        m = est[sev]
        s = stats.query(f"B == '{sev}'").squeeze()
        p = s["p-unc"]
        if isinstance(p, pd.Series):
            p = 1
        res.loc[con, sev] = (
            f"{m:.2f} ({m - iqr[sev]:.2f}-{m + iqr[sev]:.2f})"
            + ("" if (p > 0.05) else "*")
        )
    s = stats.loc[stats["p-unc"].idxmin()]
    res.loc[con, "stat"] = f"{s['U-val']:.2f}; {s['hedges']:.2f}"
    res.loc[con, "p-value"] = f"{s['p-unc']:.2e} ({s['B']})"
# For categoricals, do a Fisher's exact test
cats = meta.columns[meta.dtypes == "category"]
cats = [c for c in cats if c in keep]
for cat in cats:
    if cat == "severity_group":
        continue
    meta2 = meta.dropna(subset=[cat]).copy()
    # expected, observed, stats = pg.chi2_independence(
    #     meta.dropna(subset=[cat]), x='severity_group', y=cat)
    for val in meta[cat].dropna().unique():
        row_name = f"{cat}: {val}"
        meta2["Y"] = meta2[cat] == val
        stats = list()
        pvals = list()
        for sev in sevs:
            meta2["X"] = meta2["severity_group"] == sev
            t = pg.dichotomous_crosstab(meta2, x="X", y="Y")
            stat, p = fisher_exact(t)
            count = (meta2["X"] & meta2["Y"]).sum()
            perc = (count / meta2.shape[0]) * 100
            # res.loc[row_name, sev] = f"{count} ({perc:.2f}%) = {p:.2f}"
            res.loc[row_name, sev] = f"{count} ({perc:.2f}%)" + (
                "" if p > 0.05 else "*"
            )
            stats.append(stat)
            pvals.append(p)
        idx = np.argmin(pvals)
        res.loc[row_name, "stat"] = f"{stats[idx]:.2f}"
        res.loc[row_name, "p-value"] = f"{min(pvals):.2e} ({sevs[idx]})"
res.index.name = "Variable"
res.loc[res.index.str.startswith("race"), "negative"] = np.nan
res.loc[res.index.str.startswith("tocilizumab"), "negative"] = np.nan
res.loc[res.index.str.startswith("death"), "negative"] = np.nan
res.loc[res.index.str.startswith("hospitalization"), "negative"] = np.nan
res.loc[res.index.str.startswith("intubation"), "negative"] = np.nan
res.loc[res.index.str.startswith("diabetes"), "negative"] = np.nan
res.loc[res.index.str.startswith("hypertension"), "negative"] = np.nan
res.loc[res.index.str.startswith("obesity"), "negative"] = np.nan
res = res.loc[~res.index.str.endswith(": False"), :]

res = res.rename(
    columns={"stat": "statistic (min)", "p-value": "p-value (min)"}
)
res.index = res.index.str.capitalize()
res.columns = res.columns.str.capitalize()


statements = [
    "Statistic/P-value: Fisher’s exact test for categorical variables and Mann-Whitney U for continuous variables.",
    "Values in parenthesis represent percentages for categorical variables and interquantile range for continuous variables.",
    "The Statistic and P-value column represent the value for the class with lowest p-value (indicated in the parenthesis in the “P-value” column).",
    "* p < 0.05 for comparing each value to the “Negative” class or to “Mild” when values for “Negative” were not available.",
]

s = res.shape[0]
res.loc[np.nan, "Negative"] = np.nan
for i, text in enumerate(statements):
    res.loc[text, "Negative"] = np.nan

save_excel(
    res, output_dir / "Supplementary_Table1-auto.xlsx", "Patient data summary"
)
