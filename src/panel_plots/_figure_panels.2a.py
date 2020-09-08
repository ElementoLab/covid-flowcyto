#!/usr/bin/env python

"""

"""

import re

import pingouin as pg

from src.conf import *


output_dir = figures_dir / "panels"
output_dir.mkdir()

meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file)
matrix_red_var = pd.read_parquet(matrix_imputed_reduced_file)


# Read up various matrices that were used for fitting
meta_red = pd.read_parquet(metadata_dir / "annotation.reduced_per_patient.pq")
red_pat_early = pd.read_parquet("data/matrix_imputed_reduced.red_pat_early.pq")
red_pat_median = pd.read_parquet(
    "data/matrix_imputed_reduced.red_pat_median.pq"
)

# Demonstrate the data
# matrix = red_pat_early
# meta = meta_red
reduction = "reduced"


c = matrix.columns.str.contains
v = matrix.loc[
    :, c(r"CD45R.\+_CD\d\+\/CD.\+") & (~(c("CD62|CD95|CD197")))
].columns


neg = meta["severity_group"] == "negative"
mild = meta["severity_group"] == "mild"
seve = meta["severity_group"] == "severe"
hosp = meta["hospitalization"] == "True"
intu = meta["intubation"] == "True"
deat = meta["death"] == "dead"
meta.loc[neg, "group"] = "negative"
meta.loc[mild & (~hosp), "group"] = "mild_hospitalized-neg"
meta.loc[mild & hosp, "group"] = "mild_hospitalized"
meta.loc[seve & (~intu) & (~deat), "group"] = "severe_intubated-neg_death-neg"
meta.loc[seve & (~intu) & (deat), "group"] = "severe_intubated-neg_death"
meta.loc[seve & intu & (~deat), "group"] = "severe_intubated_death-neg"
meta.loc[seve & intu & deat, "group"] = "severe_intubated_death"

meta["group"] = pd.Categorical(
    meta["group"],
    categories=[
        "negative",
        "mild_hospitalized-neg",
        "mild_hospitalized",
        "severe_intubated-neg_death-neg",
        "severe_intubated-neg_death",
        "severe_intubated_death-neg",
        "severe_intubated_death",
    ],
    ordered=True,
)

cat_var = "group"
panel_name = "CD3_memory"

figfile = (
    output_dir
    / f"variable_illustration.{cat_var}.panel_{panel_name}.{reduction}.swarm+boxen.svg"
)
data = (
    matrix.loc[:, v]
    .join(meta[[cat_var]])
    .melt(id_vars=[cat_var], var_name="population", value_name="abundance (%)",)
)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", [colorblind[2], colorblind[1], colorblind[3]]
)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["forestgreen", "gray", "purple"]
)

n = meta[cat_var].nunique()
palette = cmap(np.arange(n + 1) / n)
kws = dict(
    data=data,
    x=cat_var,
    y="abundance (%)",
    hue=cat_var,
    palette=palette,  # "RdYlGn_r",
)
gridkws = dict(sharey=False, height=3, aspect=1, col_wrap=4)
grid = sns.FacetGrid(data=data, col="population", **gridkws)
grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
for ax in grid.axes.flat:
    for x in ax.get_children():
        if isinstance(x, patches):
            x.set_alpha(0.25)

grid.map_dataframe(sns.swarmplot, **kws)
for ax in grid.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# better title
for ax in grid.axes.flat:
    var = ax.get_title().replace("population = ", "")
    try:
        child, parent = re.findall(r"(.*)/(.*)", var)[0]
        ax.set_title(child)
        ax.set_ylabel(f"% {parent}")
    except IndexError:
        ax.set_title(var)

grid.savefig(figfile)
plt.close(grid.fig)


v = ["CD4+_CD8+/CD3+"]

cat_var = "group"
panel_name = "CD4_CD8_ratio"

figfile = (
    output_dir
    / f"variable_illustration.{cat_var}.panel_{panel_name}.{reduction}.swarm+boxen.svg"
)
data = (
    (matrix.loc[:, v])
    .join(meta[[cat_var]])
    .melt(id_vars=[cat_var], var_name="population", value_name="abundance (%)",)
)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", [colorblind[2], colorblind[1], colorblind[3]]
)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["forestgreen", "gray", "purple"]
)

n = meta[cat_var].nunique()
palette = cmap(np.arange(n + 1) / n)
kws = dict(
    data=data,
    x=cat_var,
    y="abundance (%)",
    hue=cat_var,
    palette=palette,  # "RdYlGn_r",
)
gridkws = dict(sharey=False, height=3, aspect=1, col_wrap=4)
grid = sns.FacetGrid(data=data, col="population", **gridkws)
grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
for ax in grid.axes.flat:
    for x in ax.get_children():
        if isinstance(x, patches):
            x.set_alpha(0.25)

grid.map_dataframe(sns.swarmplot, **kws)
for ax in grid.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# better title
for ax in grid.axes.flat:
    var = ax.get_title().replace("population = ", "")
    try:
        child, parent = re.findall(r"(.*)/(.*)", var)[0]
        ax.set_title(child)
        ax.set_ylabel(f"% {parent}")
    except IndexError:
        ax.set_title(var)

grid.savefig(figfile)
plt.close(grid.fig)


v = ["CD4+_CD8+/CD3+"]

cat_var = "severity_group"
panel_name = "CD4_CD8_ratio"

figfile = (
    output_dir
    / f"variable_illustration.{cat_var}.panel_{panel_name}.{reduction}.swarm+boxen.svg"
)
data = (
    (matrix.loc[:, v])
    .join(meta[[cat_var]])
    .melt(id_vars=[cat_var], var_name="population", value_name="abundance (%)",)
)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", [colorblind[2], colorblind[1], colorblind[3]]
)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["forestgreen", "gray", "purple"]
)

n = meta[cat_var].nunique()
# palette = matplotlib.colors.ListedColormap(cmap(np.arange(n + 1) / n), name="severity_group")
palette = cmap(np.arange(n + 1) / n)
kws = dict(
    data=data,
    x=cat_var,
    y="abundance (%)",
    hue=cat_var,
    palette=palette,  # "RdYlGn_r",
)
gridkws = dict(sharey=False, height=3, aspect=1, col_wrap=4)
grid = sns.FacetGrid(data=data, col="population", **gridkws)
grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
for ax in grid.axes.flat:
    for x in ax.get_children():
        if isinstance(x, patches):
            x.set_alpha(0.25)

grid.map_dataframe(sns.swarmplot, **kws)
for ax in grid.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# better title
for ax in grid.axes.flat:
    var = ax.get_title().replace("population = ", "")
    try:
        child, parent = re.findall(r"(.*)/(.*)", var)[0]
        ax.set_title(child)
        ax.set_ylabel(f"% {parent}")
    except IndexError:
        ax.set_title(var)

grid.savefig(figfile)
plt.close(grid.fig)


data = (matrix.loc[:, v].join(meta[[cat_var]])).dropna()

groups = (
    data["group"]
    .value_counts()[data["group"].value_counts() > 1]
    .index.tolist()
)
data = data.loc[data["group"].isin(groups), :]
data["group"] = data["group"].cat.remove_unused_categories()
res = pd.concat(
    [
        pg.pairwise_ttests(
            data=data, parametric=False, dv=v, between="group"
        ).assign(var=v)
        for v in data.columns[:-1]
    ]
).drop("Contrast", axis=1)
res["p-cor"] = pg.multicomp(res["p-unc"].values, method="fdr_bh")[1]
res = res.merge(pd.Series(panel, name="panel").rename_axis("var").reset_index())
res.to_csv("diff.detailed2.csv", index=False)
