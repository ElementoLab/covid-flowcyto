#!/usr/bin/env python

"""
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

import imc
from imc.graphics import to_color_series


def zscore(x, axis=0):
    return (x - x.mean(axis)) / x.std(axis)


def text(x, y, s, ax=None, **kws):
    if ax is None:
        ax = plt.gca()
    return [ax.text(x=_x, y=_y, s=_s, **kws) for _x, _y, _s in zip(x, y, s)]


def add_colorbar(im, ax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation="vertical")


# plt.text = np.vectorize(plt.text)
# matplotlib.axes._subplots.Axes.text = np.vectorize(matplotlib.axes._subplots.Axes.text)


figkws = dict(dpi=300, bbox_inches="tight")

original_dir = Path("data") / "original"
metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")
output_dir = results_dir / "supervised"
output_dir.mkdir(exist_ok=True, parents=True)

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

metadata_file = metadata_dir / "annotation.pq"
matrix_file = data_dir / "matrix.pq"
matrix_imputed_file = data_dir / "matrix.pq"


categories = ["patient", "severity_group", "intubated", "death", "heme", "bmt", "obesity"]
technical = ["date"]
continuous = ["timepoint"]
variables = categories + continuous + technical


alpha_thresh = 0.01
log_alpha_thresh = -np.log10(alpha_thresh)


meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file).sort_index(0).sort_index(1)


cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns
parent_population = cols[1].rename("parent_population")


to_fit = matrix.copy()
to_fit.columns = (
    to_fit.columns.str.replace("/", "___")
    .str.replace("+", "pos")
    .str.replace("-", "neg")
    .str.replace("(", "_O_")
    .str.replace(")", "_C_")
)

# Fit linear models

# # First try a model with only categories available for most data
cur_variables = categories[:4] + technical

# and reduce replicates by mean
meta_reduced = meta.drop_duplicates(subset=["sample_id"])
to_fit_reduced = to_fit.groupby(meta["sample_id"]).mean().set_index(meta_reduced.index)


to_fit.groupby(meta["sample_id"]).std()


_coefs = dict()
for m, d, label in [(meta, to_fit, "original"), (meta_reduced, to_fit_reduced, "reduced")]:
    data = zscore(d).join(m[cur_variables]).dropna()

    __coefs = list()
    for col in to_fit.columns:
        formula = f"{col} ~ {' + '.join(cur_variables)}"
        md = smf.glm(formula, data)
        mdf = md.fit()
        __coefs.append(
            mdf.params.to_frame(name="coef").join(mdf.pvalues.rename("pval")).assign(variable=col)
        )
    _coefs[label] = pd.concat(__coefs).rename_axis(index="comparison").drop("Intercept", axis=0)

    _coefs[label].to_csv(output_dir / f"differential.{label}.results.csv")


for label in ["original", "reduced"]:

    coefs = _coefs[label]
    long_f = coefs.pivot_table(index="variable", columns="comparison")
    long_f.index = matrix.columns

    changes = long_f["coef"]
    pvals = -np.log10(long_f["pval"])
    qvals = -np.log10(
        long_f["pval"]
        .apply(multipletests, method="fdr_bh")
        .apply(lambda x: pd.Series(x[1]))
        .T.set_index(long_f.index)
    )

    # Visualize

    # # Heatmaps
    kwargs = dict(center=0, cmap="RdBu_r", robust=True, metric="correlation")
    grid = sns.clustermap(changes, cbar_kws=dict(label="log2(fold-change)"), **kwargs)
    grid.savefig(output_dir / f"differential.{label}.lfc.all_vars.clustermap.svg")
    grid = sns.clustermap(pvals, cbar_kws=dict(label="-log10(p-value)"), **kwargs)
    grid.savefig(output_dir / f"differential.{label}.pvals_only.all_vars.clustermap.svg")

    # # # Heatmap combinin both change and significance
    cols = ~changes.columns.str.contains("|".join(technical))
    grid = sns.clustermap(
        changes.loc[:, cols],
        cbar_kws=dict(label="log2(Fold-change"),
        row_colors=pvals.loc[:, cols],
        **kwargs,
    )
    grid.savefig(output_dir / f"differential.{label}.join_lfc_pvals.all_vars.clustermap.svg")

    # # # only significatnt
    sigs = (qvals >= log_alpha_thresh).any(1)
    grid = sns.clustermap(
        changes.loc[sigs, cols],
        cbar_kws=dict(label="log2(Fold-change"),
        row_colors=pvals.loc[sigs, cols],
        xticklabels=True,
        yticklabels=True,
        **kwargs,
    )
    grid.savefig(
        output_dir / f"differential.{label}.join_lfc_pvals.p<{alpha_thresh}_only.clustermap.svg"
    )

    # # Volcano plots
    for category in categories:
        cols = pvals.columns[pvals.columns.str.contains(category)]
        n = len(cols)
        if not n:
            continue

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
        axes = axes.squeeze(0)
        for i, col in enumerate(cols):
            sigs = qvals[col] >= log_alpha_thresh
            kwargs = dict(s=2, alpha=0.5, color="grey")
            axes[i].scatter(changes.loc[~sigs, col], pvals.loc[~sigs, col], **kwargs)
            kwargs = dict(s=10, alpha=1.0, c=qvals.loc[sigs, col], cmap="Reds", vmin=0)
            im = axes[i].scatter(changes.loc[sigs, col], pvals.loc[sigs, col], **kwargs)
            name = re.findall(r"^(.*)\[", col)[0]
            inst = re.findall(r"\[T.(.*)\]", col)[0]
            # v = -np.log10(multipletests([alpha_thresh] * changes[col].shape[0])[1][0])
            v = pvals.loc[~sigs, col].max()
            axes[i].axhline(v, color="grey", linestyle="--", alpha=0.5)
            axes[i].axvline(0, color="grey", linestyle="--", alpha=0.5)
            axes[i].set_title(f"{name}: {inst}/{meta[category].min()}")
            axes[i].set_xlabel("log2(Fold-change)")
            axes[i].set_ylabel("-log10(p-value)")

            texts = text(
                changes.loc[sigs, col],
                pvals.loc[sigs, col],
                changes.loc[sigs, col].index,
                axes[i],
                fontsize=5,
            )
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black"), ax=axes[i])
            add_colorbar(im, axes[i])

        fig.savefig(output_dir / f"differential.{label}.test_{category}.volcano.svg")
        plt.close(fig)

    # # Illustration of top hits
    n_plot = 7

    for category in categories:
        cols = pvals.columns[pvals.columns.str.contains(category)]
        n = len(cols)
        if not n:
            continue
        for i, col in enumerate(cols):
            v = qvals[col].sort_values()
            sigs = v[v >= log_alpha_thresh]
            if sigs.empty:
                continue
            print(category, sigs)
            sigs = sigs.tail(n_plot).index
            data = matrix.loc[:, sigs].join(meta[category]).melt(id_vars=category)

            grid = sns.catplot(
                data=data, col="variable", y="value", x=category, sharey=False, height=3
            )
            for ax in grid.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            # grid.map(sns.boxplot)
            grid.savefig(output_dir / f"differential.{label}.test_{category}.{col}.swarm.svg")
            plt.close(grid.fig)

    # # For Mixed Effect model
    # from patsy import dmatrices

    # #%% Generate Design Matrix for later use
    # Y, X = dmatrices(formula, data=data, return_type="matrix")
    # Terms = X.design_info.column_names
    # _, Z = dmatrices("rt ~ -1+subj", data=data, return_type="matrix")
    # X = np.asarray(X)  # fixed effect
    # Z = np.asarray(Z)  # mixed effect
    # Y = np.asarray(Y).flatten()
    # nfixed = np.shape(X)
    # nrandm = np.shape(Z)

# Add lock file
open(output_dir / "__done__", "w")
