import scanpy as sc


from src.conf import *

# plt.rcParams["image.cmap"] = "viridis"

# panel_name = "WB_NK_KIR"
# panel_name = "PBMC_MDSC"
panel_name = "WB_Memory"

label = "full"

panel_dir = results_dir / "single_cell" / panel_name
processed_h5ad = panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"

a = sc.read(processed_h5ad)


x = a.to_df().join(a.obs[["sample_id", "severity_group"]])

var1 = "CD95(Pacific Blue-A)"
# var2 = "CD25(PE-A)"
x0 = x.loc[x[var1] < 0]
total = x["severity_group"].value_counts()
neg = x0["severity_group"].value_counts()

fraction = neg / total

neg_pat = (x[var1] < 0).groupby(x["sample_id"]).sum()
total_pat = x.groupby("sample_id").size()
fraction_pat = (
    meta.drop_duplicates("sample_id")
    .set_index("sample_id")[["severity_group"]]
    .join((neg_pat / total_pat).rename("fraction") * 100)
)


fig, ax = plt.subplots(1, 1, figsize=(4, 0.8))
ax.axvline(0, linestyle="--", color="grey")
sns.distplot(x[var1], ax=ax)
fig.savefig(
    figures_dir / "panels" / "Figure2.Fas_expression.distplot.svg",
    **figkws,
    tight_layout=True,
)

fig, axes = plt.subplots(1, 3, figsize=(9, 4))
axes[0].axhline(0, linestyle="--", color="grey")
sns.boxenplot(
    data=x,
    x="severity_group",
    y=var1,
    palette=palettes.get("severity_group"),
    ax=axes[0],
)
sns.barplot(
    fraction.index,
    fraction * 100,
    palette=palettes.get("severity_group"),
    ax=axes[1],
)
kws = dict(
    x="severity_group", y="fraction", palette=palettes.get("severity_group")
)
sns.boxenplot(data=fraction_pat, ax=axes[2], **kws)
for ch in axes[2].get_children():
    if isinstance(ch, patches):
        ch.set_alpha(0.25)
sns.swarmplot(data=fraction_pat, **kws, ax=axes[2])
axes[0].set(ylabel="Fas expression\n(compensated intensity)")
axes[1].set(ylabel="% Fas- cells")
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.savefig(
    figures_dir / "panels" / "Figure2.Fas_expression.svg",
    **figkws,
    tight_layout=True,
)


var2 = "CD25(PE-A)"
x0 = x.loc[x[var2] > 0]
total = x["severity_group"].value_counts()
pos = x0["severity_group"].value_counts()

fraction = pos / total

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].axhline(0, linestyle="--", color="grey")
sns.boxenplot(
    data=x,
    x="severity_group",
    y=var2,
    palette=palettes.get("severity_group"),
    ax=axes[0],
)
sns.barplot(
    fraction.index,
    fraction * 100,
    palette=palettes.get("severity_group"),
    ax=axes[1],
)
axes[0].set(ylabel=f"CD25 expression\n(compensated intensity)")
axes[1].set(ylabel=f"% CD25+ cells")
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.savefig(
    figures_dir / "panels" / "Figure2.CD25_expression.svg",
    **figkws,
    tight_layout=True,
)


var1 = "CD95(Pacific Blue-A)"
var2 = "CD25(PE-A)"

fig, axes = plt.subplots(1, 4, figsize=(3 * 4, 3))
for ax in axes.flat:
    ax.axhline(0, linestyle="--", color="grey")
    ax.axvline(0, linestyle="--", color="grey")
for ax, lab, color in zip(
    axes.flat, meta["severity_group"].cat.categories, palettes["severity_group"]
):
    y = x.loc[x["severity_group"] == lab]
    ax.scatter(y[var2], y[var1], s=1, alpha=0.25, rasterized=True, color=color)
    t = y.shape[0]
    q0 = (((y[var2] > 0) & (y[var1] > 0)).sum() / t) * 100
    ax.text(0.5, 0.5, s=f"{q0:.1f}")
    q1 = (((y[var2] > 0) & (y[var1] < 0)).sum() / t) * 100
    ax.text(0.5, -0.5, s=f"{q1:.1f}")
    q2 = (((y[var2] < 0) & (y[var1] > 0)).sum() / t) * 100
    ax.text(-0.5, 0.5, s=f"{q2:.1f}")
    q3 = (((y[var2] < 0) & (y[var1] < 0)).sum() / t) * 100
    ax.text(-0.5, -0.5, s=f"{q3:.1f}")
    ax.set(title=lab + f"\n{q0:.1f}; {q1:.1f}; {q2:.1f}; {q3:.1f}")

fig.savefig(
    figures_dir / "panels" / "Figure2.CD25_vs_Fas_expression.distplot.svg",
    **figkws,
    tight_layout=True,
)
