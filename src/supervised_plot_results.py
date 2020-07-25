#!/usr/bin/env python

"""
This script plots the outcome of the models as volcano plots, MA plots
and illustrates some examples in the raw data for each model/data combination.
"""

import re

from adjustText import adjust_text
from scipy.stats import pearsonr

from imc.graphics import to_color_series

from src.conf import *


def add_colorbar(im, ax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation="vertical")


def volcano_plot(cols, coefs, lpvals, lqvals, log_alpha_thresh):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
    axes = axes.squeeze(0)
    for i, col in enumerate(cols):
        sigs = lqvals[col] >= log_alpha_thresh
        kwargs = dict(s=2, alpha=0.5, color="grey")
        axes[i].scatter(coefs.loc[~sigs, col], lpvals.loc[~sigs, col], **kwargs)
        kwargs = dict(
            s=10, alpha=1.0, c=lqvals.loc[sigs, col], cmap="Reds", vmin=0
        )
        im = axes[i].scatter(
            coefs.loc[sigs, col], lpvals.loc[sigs, col], **kwargs
        )
        if "[" in col:
            name = re.findall(r"^(.*)\[", col)[0]
            inst = re.findall(r"\[T.(.*)\]", col)[0]
        else:
            name = inst = col

        # v = -np.log10(multipletests([alpha_thresh] * coefs[col].shape[0])[1][0])
        v = lpvals.loc[~sigs, col].max()
        axes[i].axhline(v, color="grey", linestyle="--", alpha=0.5)
        axes[i].axvline(0, color="grey", linestyle="--", alpha=0.5)
        try:
            axes[i].set_title(f"{name}: {inst}/{meta[variable].min()}")
        except TypeError:
            axes[i].set_title(f"{name}: {inst}")
        axes[i].set_xlabel("log2(Fold-change) " + r"($\beta$)")
        axes[i].set_ylabel("-log10(p-value)")

        texts = text(
            coefs.loc[sigs, col],
            lpvals.loc[sigs, col],
            coefs.loc[sigs, col].index,
            axes[i],
            fontsize=5,
        )
        adjust_text(
            texts, arrowprops=dict(arrowstyle="->", color="black"), ax=axes[i],
        )
        add_colorbar(im, axes[i])
    return fig


def ma_plot(cols, coefs, lpvals, lqvals, log_alpha_thresh):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
    axes = axes.squeeze(0)
    for i, col in enumerate(cols):
        sigs = lqvals[col] >= log_alpha_thresh
        kwargs = dict(s=2, alpha=0.5, color="grey")
        axes[i].scatter(
            coefs.loc[~sigs, "Intercept"], coefs.loc[~sigs, col], **kwargs,
        )
        kwargs = dict(
            s=10, alpha=1.0, c=lqvals.loc[sigs, col], cmap="Reds", vmin=0
        )
        im = axes[i].scatter(
            coefs.loc[sigs, "Intercept"], coefs.loc[sigs, col], **kwargs
        )
        if "[" in col:
            name = re.findall(r"^(.*)\[", col)[0]
            inst = re.findall(r"\[T.(.*)\]", col)[0]
        else:
            name = inst = col
        axes[i].axhline(0, color="grey", linestyle="--", alpha=0.5)
        axes[i].set_title(f"{name}: {inst}/{meta[variable].min()}")
        axes[i].set_xlabel("Mean")
        axes[i].set_ylabel("log2(fold-change)")

        texts = text(
            coefs.loc[sigs, "Intercept"],
            coefs.loc[sigs, col],
            coefs.loc[sigs, col].index,
            axes[i],
            fontsize=5,
        )
        adjust_text(
            texts, arrowprops=dict(arrowstyle="->", color="black"), ax=axes[i],
        )
        add_colorbar(im, axes[i])
    return fig


def swarm_boxen_plot(data, variable, col, coefs, qvals):
    categorical = data[variable].dtype

    kws = dict(data=data, x=variable, y="value", hue=variable, palette="tab10")
    grid = sns.FacetGrid(
        data=data, col="variable", sharey=False, height=3, col_wrap=4
    )
    if categorical:
        grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
    else:
        grid.map_dataframe(sns.scatterplot, **kws)

    for ax in grid.axes.flat:
        [x.set_alpha(0.25) for x in ax.get_children() if isinstance(x, patches)]
    grid.map_dataframe(sns.swarmplot, **kws)

    for ax in grid.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # add stats to title
    group = re.findall(r"^.*\[T.(.*)\]", col)[0] if "[" in col else col
    for ax in grid.axes.flat:
        var = ax.get_title().replace("variable = ", "")
        pop = var
        try:
            pop, parent = re.findall(r"(.*)/(.*)", pop)[0]
            ax.set_ylabel(f"% {parent}")
        except IndexError:
            pass
        ax.set_title(
            pop
            + f"\n{group}/{control}:\n"
            + f"Coef = {coefs.loc[var, col]:.3f}; "
            + f"FDR = {qvals.loc[var, col]:.3e}"
        )
    return grid


output_dir = results_dir / "supervised"
output_dir.mkdir(exist_ok=True, parents=True)

meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file).sort_index(0).sort_index(1)
matrix_red_var = (
    pd.read_parquet(matrix_imputed_reduced_file).sort_index(0).sort_index(1)
)

alpha_thresh = 0.05
log_alpha_thresh = -np.log10(alpha_thresh)


# to annotate variables
cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns
parent_population = cols[1].rename("parent_population")

# Decide if using all samples (including technical replicates or reduced version)
# This is a reduced version, where replicates are averaged
meta_reduced = meta.drop_duplicates(subset=["sample_id"]).sort_values(
    "sample_id"
)
matrix_reduced = (
    matrix_red_var.groupby(meta["sample_id"])
    .mean()
    .set_index(meta_reduced.index)
)


# Read up various matrices that were used for fitting
meta_red = pd.read_parquet(metadata_dir / "annotation.reduced_per_patient.pq")
red_pat_early = pd.read_parquet("data/matrix_imputed_reduced.red_pat_early.pq")
red_pat_median = pd.read_parquet(
    "data/matrix_imputed_reduced.red_pat_median.pq"
)


# Compare models (original vs reduced data)
reducts = [
    ("original", "reduced"),
    ("reduced", "reduced_early"),
    ("reduced", "reduced_median"),
]

k = dict(index_col=0)
for a, b in reducts:
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(3 * len(models), 3 * 2),
        # sharex="row",
        # sharey="row",
    )
    for i, (model_name, _) in enumerate(models.items()):
        prefix = output_dir / f"differential.{model_name}."
        x = pd.read_csv(prefix + f"{a}.results.csv", **k).drop("Intercept")
        y = pd.read_csv(prefix + f"{b}.results.csv", **k).drop("Intercept")

        assert (x.index == y.index).all()
        close = np.allclose(x["coef"], y["coef"])
        kw = dict(linestyle="--", color="grey", alpha=0.2)
        kws = dict(s=2, alpha=0.5, rasterized=True)

        cx = x["coef"].clip(-20, 20)
        cy = y["coef"].clip(-20, 20)
        c = pd.concat([cx, cy], 1)
        cdna = c.dropna()
        lpx = log_pvalues(x["pval"])  # , clip=0.95
        lpy = log_pvalues(y["pval"])  # , clip=0.95
        lp = pd.concat([lpx, lpy], 1)
        lpdna = lp.dropna()

        # axes[0, i].scatter(np.tanh(x["coef"]), np.tanh(y["coef"]), **kws)
        cv = max(cx.abs().max(), cy.abs().max())
        mcv = -((1 / 8) * cv)
        pv = max(lpx.max(), lpy.max())
        cv += cv * 0.1
        pv += pv * 0.1

        axes[0, i].plot((-cv, cv), (-cv, cv), **kw)
        axes[1, i].plot((mcv, pv), (mcv, pv), **kw)

        axes[0, i].scatter(cx, cy, c=lp.mean(1), **kws)
        axes[1, i].scatter(lpx, lpy, c=c.abs().mean(1), **kws)
        r, p = pearsonr(cdna.iloc[:, 0], cdna.iloc[:, 1])
        axes[0, i].set(
            title=model_name + f"\nr = {r:.3f}, p = {p:.2e}",
            xlim=(-cv, cv),
            ylim=(-cv, cv),
        )
        r, p = pearsonr(lpdna.iloc[:, 0], lpdna.iloc[:, 1])
        axes[1, i].set(
            title=f"r = {r:.3f}, p = {p:.2e}",
            xlabel=a.capitalize(),
            xlim=(mcv, pv),
            ylim=(mcv, pv),
        )
    axes[0, 0].set(ylabel=f"Coefficient\n{b.capitalize()}")
    axes[1, 0].set(ylabel=f"-log10(P-value)\n{b.capitalize()}")
    fig.savefig(
        output_dir / f"differential.model_comparison.{a}_vs_{b}.svg", **figkws
    )
    plt.close(fig)


# Plot outcomes
# # In this case, I will use the stats from the models fitted on the largest
# # sample sizes, *but plot* the data with one dot per patient
meta_c, matrix_c, reduction = (meta_red, red_pat_median, "reduced")
per_panel = True

for i, (model_name, model) in enumerate(list(models.items())):
    prefix = f"differential.{model_name}.{reduction}."
    res = pd.read_csv(
        output_dir / f"differential.{model_name}.{reduction}.results.csv"
    )
    res = res.loc[res["llf"] < np.inf]

    long_f = res.pivot_table(index="variable", columns="comparison")
    # drop columns with levels not estimated
    long_f = long_f.loc[
        :,
        long_f.columns.get_level_values(1).isin(
            long_f["coef"].columns[long_f["coef"].abs().sum() > 1e-10]
        ),
    ]

    coefs = long_f["coef"]

    for variable in model["continuous"]:
        coefs[variable] = coefs[variable] * (meta[variable].mean() / 2)
    pvals = long_f["pval"]
    qvals = long_f["qval"]
    lpvals = long_f["log_pval"]
    lqvals = long_f["log_qval"]

    # Visualize

    # # Heatmaps
    ks = dict(center=0, cmap="RdBu_r", robust=True, metric="correlation")
    grid = sns.clustermap(coefs, cbar_kws=dict(label="log2(fold-change)"), **ks)
    grid.savefig(output_dir / prefix + "lfc.all_vars.clustermap.svg")
    plt.close(grid.fig)
    grid = sns.clustermap(lpvals, cbar_kws=dict(label="-log10(p-value)"), **ks)
    grid.savefig(output_dir / prefix + "pvals_only.all_vars.clustermap.svg")
    plt.close(grid.fig)

    # # # Heatmap combining both change and significance
    cols = ~coefs.columns.str.contains("|".join(TECHNICAL))
    grid = sns.clustermap(
        coefs.loc[:, cols].drop("Intercept", axis=1),
        cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
        row_colors=lpvals.loc[:, cols],
        xticklabels=True,
        yticklabels=True,
        **ks,
    )
    grid.savefig(output_dir / prefix + "join_lfc_pvals.all_vars.clustermap.svg")
    plt.close(grid.fig)

    # # # only significatnt
    sigs = (lqvals >= log_alpha_thresh).any(1)
    grid = sns.clustermap(
        coefs.loc[sigs, cols].drop("Intercept", axis=1),
        cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
        row_colors=lpvals.loc[sigs, cols],
        xticklabels=True,
        yticklabels=True,
        **ks,
    )
    grid.savefig(
        output_dir / prefix
        + f"join_lfc_pvals.p<{alpha_thresh}_only.clustermap.svg"
    )
    plt.close(grid.fig)

    # # Volcano plots
    for variable in model["covariates"]:
        cols = lpvals.columns[lpvals.columns.str.contains(variable)]
        n = len(cols)
        if not n:
            continue
        fig = volcano_plot(cols, coefs, lpvals, lqvals, log_alpha_thresh)
        fig.savefig(
            output_dir / prefix + f"test_{variable}.volcano.svg", **figkws
        )
        plt.close(fig)
        if per_panel:
            for panel_name, variables in panels.items():
                fig = volcano_plot(
                    cols,
                    coefs.reindex(variables).dropna(),
                    lpvals.reindex(variables).dropna(),
                    lqvals.reindex(variables).dropna(),
                    log_alpha_thresh,
                )
                fig.savefig(
                    output_dir / prefix
                    + f"test_{variable}.volcano.{panel_name}.svg",
                    **figkws,
                )
                plt.close(fig)

    # # MA plots
    for variable in model["covariates"]:
        cols = lpvals.columns[lpvals.columns.str.contains(variable)]
        n = len(cols)
        if not n:
            continue
        fig = ma_plot(cols, coefs, lpvals, lqvals, log_alpha_thresh)
        fig.savefig(
            output_dir / prefix + f"test_{variable}.maplots.svg", **figkws
        )
        plt.close(fig)
        if per_panel:
            for panel_name, variables in panels.items():
                fig = ma_plot(
                    cols,
                    coefs.reindex(variables).dropna(),
                    lpvals.reindex(variables).dropna(),
                    lqvals.reindex(variables).dropna(),
                    log_alpha_thresh,
                )
                fig.savefig(
                    output_dir / prefix
                    + f"test_{variable}.maplots.{panel_name}.svg",
                    **figkws,
                )
                plt.close(fig)

    if "interaction" in model_name:
        continue

    # # Illustration of top hits
    n_plot = 10
    for variable in model["categories"]:  # "variables"
        cols = pvals.columns[pvals.columns.str.contains(variable)]
        control = meta_c[variable].min()
        n = len(cols)
        if not n:
            continue
        for i, col in enumerate(cols):
            v = lqvals[col].sort_values()
            sigs = v  # v[v >= log_alpha_thresh] # <- actually filter
            if sigs.empty:
                continue
            # print(variable, sigs)
            sigs = sigs.tail(n_plot).index[::-1]
            data = matrix_c[sigs].join(meta_c[variable]).melt(id_vars=variable)
            # for the interaction models ->
            # data = matrix_c[sigs].join(meta[['severity_group',  variable]]).query("severity_group == 'severe'").drop("severity_group", 1).melt(id_vars=variable)
            grid = swarm_boxen_plot(data, variable, col, coefs, qvals)
            grid.savefig(
                output_dir / prefix + f"test_{variable}.{col}.swarm.svg"
            )
            plt.close(grid.fig)
            if per_panel:
                for panel_name, variables in panels.items():
                    sigs = (
                        lqvals.reindex(variables)
                        .dropna()[col]
                        .sort_values()
                        .tail(n_plot)
                        .index[::-1]
                    )
                    data = (
                        matrix_c[sigs]
                        .join(meta_c[variable])
                        .melt(id_vars=variable)
                    )
                    grid = swarm_boxen_plot(data, variable, col, coefs, qvals)
                    grid.savefig(
                        output_dir / prefix
                        + f"test_{variable}.{col}.swarm.{panel_name}.svg"
                    )
                    plt.close(grid.fig)


# # Illustrate top hits for interaction models
n_plot = 20
interactions = ["severity_group", "intubation", "death", "hospitalization"]


meta_c, matrix_c, reduction = (meta_red, red_pat_median, "reduced")

for model_name, model in {
    k: v for k, v in models.items() if "interaction" in k
}.items():
    res = pd.read_csv(
        output_dir / f"differential.{model_name}.{reduction}.results.csv",
        index_col=0,
    )
    prefix = f"differential.{model_name}.{reduction}.interaction_sex."
    r = res.sort_values("pval")
    r = r.loc[r.index.str.contains(":")]
    r = r.loc[r["coef"].abs() < 6]
    sigs = r.head(n_plot)["variable"]

    variable = [x for x in model["covariates"] if x != "sex"][0]
    d = (
        meta_c[[variable, "sex"]]
        .join(matrix_c[sigs])
        .melt(id_vars=[variable, "sex"])
        .dropna()
    )

    for cat in d.columns[d.dtypes == "category"]:
        d[cat] = d[cat].cat.remove_unused_categories()
    # grid = sns.catplot(data=d, x=variable, y='value', col='variable', col_wrap=4, hue="sex", sharey=False)

    kws = dict(data=d, x=variable, y="value", hue="sex", palette="tab10")
    grid = sns.FacetGrid(
        data=d, col="variable", sharey=False, height=3, col_wrap=4
    )
    grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=True, **kws)

    for ax in grid.axes.flat:
        [x.set_alpha(0.25) for x in ax.get_children() if isinstance(x, patches)]
    grid.map_dataframe(sns.swarmplot, dodge=True, **kws)

    for ax in grid.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # add stats to title
    # group = re.findall(r"^.*\[T.(.*)\]", col)[0] if "[" in col else col
    for ax in grid.axes.flat:
        var = ax.get_title().replace("variable = ", "")
        pop = var
        try:
            pop, parent = re.findall(r"(.*)/(.*)", pop)[0]
            ax.set_ylabel(f"% {parent}")
        except IndexError:
            pass

        s = res.loc[res["variable"] == var].drop(["Intercept"])
        s = s.loc[s.index.str.contains(":")]
        m = s["coef"].abs().argmax()
        pos = s.iloc[m].name
        c = s.iloc[m]["coef"]
        control = meta[variable].min()
        ax.set_title(
            pop
            + f"\n{pos}:\n"
            + f"Coef = {c:.3f}; "
            + f"FDR = {s['qval'].min():.3e}"
        )

    # grid.map(sns.boxplot)
    grid.savefig(output_dir / prefix + f"{variable}.swarm+boxenplot.svg")
    plt.close(grid.fig)
