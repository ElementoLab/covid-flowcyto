import string

import scanpy as sc
from mpl_toolkits.mplot3d import Axes3D

from src.conf import *


output_dir = figures_dir / "panels"


panel_name = "WB_Treg"
label = "full"

panel_dir = results_dir / "single_cell" / panel_name
processed_h5ad = panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"

a = sc.read(processed_h5ad)
x = a.to_df().join(a.obs)


# sns.distplot(x["CD4(PE-Cy7-A)"])
# sns.distplot(x["CD25(PE-A)"])


# sns.jointplot(x["CD4(PE-Cy7-A)"], x["CD25(PE-A)"])


p = "CD4(PE-Cy7-A)"
q = "CD25(PE-A)"
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(x[p], x[q], s=0.5, alpha=0.01, rasterized=True)
ax.axhline(0.0, linestyle="--", color="grey")
ax.axhline(0.7, linestyle="--", color="grey")
ax.axhline(0.9, linestyle="--", color="grey")
ax.axvline(0.0, linestyle="--", color="grey")
ax.axvline(0.8, linestyle="--", color="grey")
ax.set(xlabel=p, ylabel=q)

ax.text(-0.5, -0.5, s="A")
ax.text(-0.5, 0.5, s="B")
ax.text(-0.5, 0.75, s="C")
ax.text(0.5, -0.5, s="D")
ax.text(0.5, 0.5, s="E")
ax.text(0.5, 0.75, s="F")
ax.text(1.1, -0.5, s="G")
ax.text(1.1, 0.5, s="H")
ax.text(1.1, 1.0, s="I")
fig.savefig(output_dir / "Treg.CD4_vs_CD25.svg", **figkws)

a = (x[p] < 0) & (x[q] < 0)
b = (x[p] < 0) & (x[q] > 0) & (x[q] < 0.7)
c = (x[p] < 0) & (x[q] > 0.7)
d = (x[p] > 0) & (x[p] < 0.8) & (x[q] < 0)
e = (x[p] > 0) & (x[p] < 0.8) & (x[q] > 0) & (x[q] < 0.7)
f = (x[p] > 0) & (x[p] < 0.8) & (x[q] > 0.7)
g = (x[p] > 0.8) & (x[q] < 0)
h = (x[p] > 0.8) & (x[q] > 0) & (x[q] < 0.9)
i = (x[p] > 0.8) & (x[q] > 0.9)

# count cells
totals = x["sample_id"].value_counts()
letters = list(string.ascii_lowercase[:9])
res = pd.DataFrame(index=totals.index, columns=letters, dtype="int")
for let in letters:
    x2 = x.loc[locals()[let]]
    for sample in x["sample_id"].unique():
        # x3 = x2.loc[x2['sample_id'] == sample]
        res.loc[sample, let] = (x2["sample_id"] == sample).sum()

res_p = (res.T / totals).T * 100
res_a = res_p.join(meta.set_index("sample_id")["severity_group"])
means = res_a.groupby("severity_group").mean().T


# count cells
totals = x["sample_id"].value_counts()
letters = list(string.ascii_lowercase[:9])
cd127_exp = pd.DataFrame(index=totals.index, columns=letters, dtype="int")
for let in letters:
    x2 = x.loc[locals()[let]]
    for sample in x["sample_id"].unique():
        # x3 = x2.loc[x2['sample_id'] == sample]
        cd127_exp.loc[sample, let] = x2.loc[
            x2["sample_id"] == sample, "CD127(APC-A)"
        ].mean()
cd127_exp = cd127_exp.join(meta.set_index("sample_id")["severity_group"])


fig, ax = plt.subplots(1, 2, figsize=(9, 4))
sns.heatmap(means, cbar_kws=dict(label="% total per patient"), ax=ax[0])
sns.heatmap(
    zscore(means.T).T,
    cbar_kws=dict(label="% total per patient\n(Z-scores)"),
    ax=ax[1],
    cmap="RdBu_r",
    center=0,
)
fig.savefig(
    output_dir / "Treg.CD4_vs_CD25.population_quantification_mean.heatmap.svg",
    **figkws,
)


grid = sns.catplot(
    data=res_a.melt(id_vars="severity_group", value_name="% cells"),
    x="severity_group",
    y="% cells",
    col="variable",
    sharey=False,
    kind="bar",
    palette=palettes.get("severity_group"),
    col_wrap=3,
    aspect=1,
    height=3,
)
grid.fig.savefig(
    output_dir / "Treg.CD4_vs_CD25.population_quantification_mean.barplot.svg",
    **figkws,
)


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.barplot(
    data=cd127_exp.melt(
        id_vars="severity_group", value_name="CD127 expression"
    ),
    x="variable",
    y="CD127 expression",
    ax=ax,
    palette="inferno",
)
fig.savefig(
    output_dir / "Treg.CD4_vs_CD25.CD127_expression.barplot.svg", **figkws,
)


grid = sns.catplot(
    data=cd127_exp.melt(
        id_vars="severity_group", value_name="CD127 expression"
    ),
    x="severity_group",
    y="CD127 expression",
    col="variable",
    sharey=False,
    kind="bar",
    palette=palettes.get("severity_group"),
    col_wrap=3,
    aspect=1,
    height=3,
)
grid.fig.savefig(
    output_dir / "Treg.CD4_vs_CD25.CD127_expression_per_severity.barplot.svg",
    **figkws,
)


grid = sns.FacetGrid(
    data=res_a.melt(id_vars="severity_group", value_name="% cells"),
    col="variable",
    sharey=False,
    col_wrap=3,
    aspect=1,
    height=3,
)
grid.map_dataframe(
    sns.boxenplot,
    data=res_a.melt(id_vars="severity_group", value_name="% cells"),
    x="severity_group",
    y="% cells",
    palette=palettes.get("severity_group"),
)
for ax in grid.axes.flat:
    for x in ax.get_children():
        if isinstance(x, patches):
            x.set_alpha(0.25)

grid.map_dataframe(
    sns.swarmplot,
    data=res_a.melt(id_vars="severity_group", value_name="% cells"),
    x="severity_group",
    y="% cells",
    palette=palettes.get("severity_group"),
)
for ax in grid.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

grid.fig.savefig(
    output_dir
    / "Treg.CD4_vs_CD25.population_quantification.swarm_boxenplot.svg",
    **figkws,
)
