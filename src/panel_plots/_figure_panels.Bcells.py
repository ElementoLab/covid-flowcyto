import scanpy as sc

from src.conf import *

panel_name = "WB_IgG_IgM"
label = "full"

panel_dir = results_dir / "single_cell" / panel_name
processed_h5ad = panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"

a = sc.read(processed_h5ad)
x = a.to_df().join(a.obs)


cd19 = "CD19(Pacific Blue-A)"
cd20 = "CD20(APC-H7-A)"

igm = "sIgM(APC-A)"
igg = "sIgG(FITC-A)"


top_top = (x[cd19] > 0.925) & (x[cd20] > 0.8)
bot_right = (x[cd19] > 0.925) & (x[cd20] < 0)
bot_left = (x[cd19] < 0.925) & (x[cd20] < 0)
lef = (x[cd19] < 0.925) & (x[cd20] > 0)
rig = (x[cd19] > 0.925) & (x[cd20] > 0) & (x[cd20] < 0.8)

tot = x["severity_group"].value_counts()
top_top_f = (top_top.groupby(x["severity_group"]).sum() / tot) * 100
bot_right_f = (bot_right.groupby(x["severity_group"]).sum() / tot) * 100
bot_left_f = (bot_left.groupby(x["severity_group"]).sum() / tot) * 100
lef_f = (lef.groupby(x["severity_group"]).sum() / tot) * 100
rig_f = (rig.groupby(x["severity_group"]).sum() / tot) * 100


quant = pd.DataFrame(
    [top_top_f, bot_right_f, bot_left_f, lef_f, rig_f],
    index=["top_top_f", "bot_right_f", "bot_left_f", "lef_f", "rig_f"],
    columns=meta["severity_group"].cat.categories,
)

fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
pal = palettes.get("severity_group")
for ax, cla, col in zip(axes, meta["severity_group"].cat.categories, pal):
    _x = x.query(f"severity_group == '{cla}'")
    ax.hexbin(_x[cd19], _x[cd20], bins="log", cmap="bone_r", rasterized=True)
    ax.set(title=cla, xlabel="CD19 expression", ylabel="CD20 expression")
    for p in [0.925]:
        ax.axvline(p, linestyle="--", color="grey", linewidth=0.5)
    for p in [0, 0.8]:
        ax.axhline(p, linestyle="--", color="grey", linewidth=0.5)

    ax.text(0.7, -0.5, f"{quant.loc['bot_left_f', cla]:.2f}%", ha="center")
    ax.text(0.7, 0.5, f"{quant.loc['lef_f', cla]:.2f}%", ha="center")
    ax.text(1.15, -0.5, f"{quant.loc['bot_right_f', cla]:.2f}%", ha="center")
    ax.text(1.15, 0.5, f"{quant.loc['rig_f', cla]:.2f}%", ha="center")
    ax.text(1.15, 1.0, f"{quant.loc['top_top_f', cla]:.2f}%", ha="center")
fig.tight_layout()
fig.savefig(
    figures_dir / "panels" / "Bcells.CD19_vs_CD20.hexbin.svg", **figkws,
)
fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
pal = palettes.get("severity_group")
for ax, cla, col in zip(axes, meta["severity_group"].cat.categories, pal):
    _x = x.query(f"severity_group == '{cla}'")
    ax.scatter(_x[cd19], _x[cd20], s=0.1, alpha=0.1, color=col, rasterized=True)
    ax.set(title=cla, xlabel="CD19 expression", ylabel="CD20 expression")
    for p in [0.925]:
        ax.axvline(p, linestyle="--", color="grey", linewidth=0.5)
    for p in [0, 0.8]:
        ax.axhline(p, linestyle="--", color="grey", linewidth=0.5)

    ax.text(0.7, -0.5, f"{quant.loc['bot_left_f', cla]:.2f}%", ha="center")
    ax.text(0.7, 0.5, f"{quant.loc['lef_f', cla]:.2f}%", ha="center")
    ax.text(1.15, -0.5, f"{quant.loc['bot_right_f', cla]:.2f}%", ha="center")
    ax.text(1.15, 0.5, f"{quant.loc['rig_f', cla]:.2f}%", ha="center")
    ax.text(1.15, 1.0, f"{quant.loc['top_top_f', cla]:.2f}%", ha="center")
fig.tight_layout()
fig.savefig(
    figures_dir / "panels" / "Bcells.CD19_vs_CD20.scatter.svg", **figkws,
)


sns.catplot(
    data=quant.reset_index().melt(id_vars=["index"]),
    col="index",
    y="value",
    x="variable",
    sharey=False,
    kind="bar",
    palette=pal,
)

sns.violinplot(data=x.loc[top], x="severity_group", y=igm, palette=pal)
sns.violinplot(data=x.loc[bot], x="severity_group", y=igm, palette=pal)


for p, n in zip(
    [top_top, bot_right, bot_left, lef, rig],
    ["top_top", "bot_right", "bot_left", "lef", "rig"],
):
    x.loc[p, "loc"] = n

markers = [
    "sIgG(FITC-A)",
    "CD25(PE-A)",
    "CD27(PerCP-Cy5-5-A)",
    "CD10(PE-Cy7-A)",
    "sIgM(APC-A)",
    "CD5(BV605-A)",
]
for marker in markers:
    # # fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    # grid = sns.catplot(
    #     data=x[["severity_group", "loc", marker]],
    #     col="loc",
    #     y=marker,
    #     x="severity_group",
    #     palette=pal,
    #     ax=ax,
    #     kind="bar",
    #     sharey=False,
    #     aspect=1.2,
    #     height=2,
    # )
    # # fig.tight_layout()
    # grid.fig.savefig(
    #     figures_dir
    #     / "panels"
    #     / f"Bcells.further_gated.{marker}_expression.barplot.svg",
    #     **figkws,
    # )

    x.loc[:, "mpos"] = x[marker] > 0
    # fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    grid = sns.catplot(
        data=x[["severity_group", "loc", "mpos"]],
        col="loc",
        y="mpos",
        x="severity_group",
        palette=pal,
        ax=ax,
        kind="bar",
        sharey=False,
        aspect=1.2,
        height=2,
    )
    for ax in grid.axes.flat:
        ax.set_ylim(0, 1)
    # ax.set(ylabel="Fraction of positive")
    # grid.fig.tight_layout()
    grid.fig.savefig(
        figures_dir
        / "panels"
        / f"Bcells.further_gated.{marker}_expression.fraction_pos.barplot.svg",
        **figkws,
    )

    # # per patient

    # size= x.groupby(['patient_code', 'severity_group', 'loc']).size()
    # pos = x.groupby(['patient_code', 'severity_group', 'loc'])['mpos'].sum().dropna()

    # sns.swarmplot(data=((pos / size).dropna() * 100).reset_index(), x='loc', y=0, hue='severity_group')
