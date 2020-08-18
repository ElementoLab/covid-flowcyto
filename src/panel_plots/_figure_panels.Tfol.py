import string

import scanpy as sc
from mpl_toolkits.mplot3d import Axes3D

from src.conf import *


output_dir = figures_dir / "panels"


panel_name = "WB_T3"
label = "full"

panel_dir = results_dir / "single_cell" / panel_name
processed_h5ad = panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"

a = sc.read(processed_h5ad)
x = a.to_df().join(a.obs)


# sns.distplot(x["CD4(BV605-A)"])
# sns.distplot(x["CD185(PE-A)"])
# sns.distplot(x["CD278(PerCP-Cy5-5-A)"])

# sns.jointplot(x["CD4(BV605-A)"], x["CD185(PE-A)"])

p = "CD185(PE-A)"
q = "CD278(PerCP-Cy5-5-A)"
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(x[p], x[q], s=0.5, alpha=0.01, rasterized=True)
ax.axhline(0.0, linestyle="--", color="grey")
ax.axvline(0.0, linestyle="--", color="grey")
ax.axvline(0.75, linestyle="--", color="grey")
ax.set(xlabel=p, ylabel=q)

ax.text(-0.5, -0.5, s="A")
ax.text(-0.5, 0.75, s="B")
ax.text(0.5, -0.5, s="C")
ax.text(0.5, 0.75, s="D")
ax.text(0.9, -0.5, s="E")
ax.text(0.9, 0.75, s="F")
fig.savefig(output_dir / "Tfol.CD185_vs_PD1.svg", **figkws)

a = (x[p] < 0) & (x[q] < 0)
b = (x[p] < 0) & (x[q] > 0)
c = (x[p] > 0) & (x[q] < 0)
d = (x[p] > 0) & (x[p] > 0.9) & (x[q] > 0)
e = (x[p] > 0.9) & (x[q] < 0)
f = (x[p] > 0) & (x[q] > 0.9)


totals = x["sample_id"].value_counts()
letters = list(string.ascii_lowercase[:6])
res = pd.DataFrame(index=totals.index, columns=letters, dtype="int")
for let in letters:
    x2 = x.loc[locals()[let]]
    for sample in x["sample_id"].unique():
        # x3 = x2.loc[x2['sample_id'] == sample]
        res.loc[sample, let] = (x2["sample_id"] == sample).sum()


res_p = (res.T / totals).T * 100
res_a = res_p.join(meta.set_index("sample_id")["severity_group"])
means = res_a.groupby("severity_group").mean().T


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
    output_dir / "Tfol.CD185_vs_PD1.population_quantification_mean.heatmap.svg",
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
    output_dir / "Tfol.CD185_vs_PD1.population_quantification_mean.barplot.svg",
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
    / "Tfol.CD185_vs_PD1.population_quantification.swarm_boxenplot.svg",
    **figkws,
)
