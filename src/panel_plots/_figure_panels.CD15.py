import scanpy as sc

from src.conf import *

panel_name = "PBMC_MDSC"
label = "full"

panel_dir = results_dir / "single_cell" / panel_name
processed_h5ad = panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"

a = sc.read(processed_h5ad)
x = a.to_df().join(a.obs)

fig, axes = plt.subplots(
    1,
    3 * 2,
    figsize=(3 * 3 * 2, 1 * 4),
    gridspec_kw=dict(width_ratios=[0.2, 0.1] * 3),
)
pal = palettes.get("severity_group")
for cla, col in zip(meta["severity_group"].cat.categories, pal):
    _x = x.query(f"severity_group == '{cla}'")["CD15(FITC-A)"]
    sns.distplot(_x, hist=False, color=col, ax=axes[0])
    axes[0].set(title="All CD16+ cells", xlabel="CD15 expression")
    axes[0].axvline(0, linestyle="--", color="grey")
frac = (x["CD15(FITC-A)"] > 0).groupby(x["severity_group"]).sum() / x[
    "severity_group"
].value_counts()
sns.barplot(frac * 100, frac.index, palette=pal, ax=axes[1])
axes[1].set(xlabel="% CD15 positive")
axes[1].set_yticklabels([])

x2 = x.loc[x["CD3(PE-Cy7-A)"] < 0]
for cla, col in zip(meta["severity_group"].cat.categories, pal):
    _x = x2.query(f"severity_group == '{cla}'")["CD15(FITC-A)"]
    sns.distplot(_x, hist=False, color=col, ax=axes[2])
    axes[2].set(title="CD16+, CD3- cells", xlabel="CD15 expression")
    axes[2].axvline(0, linestyle="--", color="grey")
frac = (x2["CD15(FITC-A)"] > 0).groupby(x2["severity_group"]).sum() / x2[
    "severity_group"
].value_counts()
sns.barplot(frac * 100, frac.index, palette=pal, ax=axes[3])
axes[3].set(xlabel="% CD15 positive")
axes[3].set_yticklabels([])

x3 = x.loc[(x["CD3(PE-Cy7-A)"] < 0) & (x["CD33(PE-A)"] > 0)]
for cla, col in zip(meta["severity_group"].cat.categories, pal):
    _x = x3.query(f"severity_group == '{cla}'")["CD15(FITC-A)"]
    sns.distplot(_x, hist=False, color=col, ax=axes[4])
    axes[4].set(title="CD16+, CD3-, CD33+ cells", xlabel="CD15 expression")
    axes[4].axvline(0, linestyle="--", color="grey")
frac = (x3["CD15(FITC-A)"] > 0).groupby(x3["severity_group"]).sum() / x3[
    "severity_group"
].value_counts()
sns.barplot(frac * 100, frac.index, palette=pal, ax=axes[5])
axes[5].set(xlabel="% CD15 positive")
axes[5].set_yticklabels([])
fig.tight_layout()
fig.savefig(
    figures_dir / "panels" / "Figure2.CD5_expression_positivity.svg", **figkws,
)


from scipy.stats import fisher_exact
import pingouin as pg  # type: ignore

pg.chi2_independence(data=x, x="severity_group", y="cluster")

y = x[["cluster"]].join(pd.get_dummies(x["severity_group"]))
for cat in x["severity_group"].unique():
    pg.chi2_independence(data=y, x="cluster", y=cat)


y = x[["severity_group"]].join(pd.get_dummies(x["cluster"]))
v = dict()
for cat in x["cluster"].unique():
    v[cat] = pg.chi2_independence(data=y, x="severity_group", y=cat)[2].iloc[-1]


y = pd.get_dummies(x[["severity_group", "cluster"]])
for seve in meta["severity_group"].cat.categories:
    for clus in x["cluster"].unique():
        y2 = y[["severity_group_" + seve, "cluster_" + clus]]
        tab = pg.dichotomous_crosstab(
            data=y2, x="severity_group_" + seve, y="cluster_" + clus
        )
        fisher_exact(tab.iloc[[1, 0], [1, 0]])
