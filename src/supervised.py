#!/usr/bin/env python

"""
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
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


def rename_back(x: str) -> str:
    return (
        pd.Series(x)
        .str.replace("___", "/")
        .str.replace("pos", "+")
        .str.replace("neg", "-")
        .str.replace("_O_", "(")
        .str.replace("_C_", ")")
    )[0]


def log_pvalues(x, f=0.1):
    """
    Calculate -log10(p-value) of array.

    Replaces infinite values with:

    .. highlight:: python
    .. code-block:: python

        max(x) + max(x) * f

    that is, fraction ``f`` more than the maximum non-infinite -log10(p-value).

    Parameters
    ----------
    x : :class:`pandas.Series`
        Series with numeric values
    f : :obj:`float`
        Fraction to augment the maximum value by if ``x`` contains infinite values.

        Defaults to 0.1.

    Returns
    -------
    :class:`pandas.Series`
        Transformed values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = -np.log10(x)
        rmax = ll[ll != np.inf].max()
        return ll.replace(np.inf, rmax + rmax * f)


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


categories = ["patient", "severity_group", "intubated", "death"]  # , "heme", "bmt", "obesity"]
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


facs = matrix.copy()
facs.columns = (
    facs.columns.str.replace("/", "___")
    .str.replace("+", "pos")
    .str.replace("-", "neg")
    .str.replace("(", "_O_")
    .str.replace(")", "_C_")
)

# This is a reduced version, where replicates are averaged
meta_reduced = meta.drop_duplicates(subset=["sample_id"]).sort_values("sample_id")
facs_reduced = facs.groupby(meta["sample_id"]).mean().set_index(meta_reduced.index)


# Fit linear models
#
#    Here we have a few issues and a few options for each:
#     - design:
#         - controls were sampled one or more times while cases only once:
#             - reduce controls by mean? -> can't model batch
#             - add patient as mixed effect? -> don't have more than one sample for cases
#     - missing data:
#         - imputation of continuous values only ~0.1% missing so, no brainer
#         - imputation of categoricals?
#     - proportion nature of the data:
#         - z-score (loose sensitivity, harder to interpret coefficients)
#         - logistic reg (did not converge for many cases :()
#         - use Gamma GLM + log link (ok, but large coefficients sometimes :/)
#         - use Binomial GLM (no power?)
#

res = dict()

# # First try a model with only categories available for most data
model_name = "categoricals"
cur_variables = categories + technical

for m, d, label, fit_vars in [
    (meta, facs, "original", cur_variables),
    (meta_reduced, facs_reduced, "reduced", cur_variables[:-1]),
]:
    # data = zscore(d).join(m[fit_vars]).dropna()
    data = d.join(m[fit_vars]).dropna()

    _res = list()
    for col in tqdm(d.columns):
        # data[col] = data[col] / 100  # for logit or binomial

        formula = f"{col} ~ {' + '.join(fit_vars)}"
        # formula = f"{col} ~ severity_group"
        # md = smf.glm(formula, data)
        md = smf.glm(formula, data, family=sm.families.Gamma(sm.families.links.log()))
        # md = smf.logit(formula, data)
        # md = smf.glm(formula, data, family=sm.families.Binomial())

        # mdf = md.fit(maxiter=100)
        mdf = md.fit_regularized(maxiter=100, refit=True)  # , L1_wt=1 # <- Ridge
        params = pd.Series(mdf.params, index=md.exog_names, name="coef")
        pvalues = pd.Series(mdf.pvalues, index=md.exog_names, name="pval")

        # fig, ax = plt.subplots()
        # sns.boxplot(data=data, x="severity_group", y=col, ax=ax)
        # sns.swarmplot(data=data, x="severity_group", y=col, ax=ax)
        # ax.set_title(str(params[params.index.str.contains("severity_group")]))

        _res.append(params.to_frame().join(pvalues).assign(variable=rename_back(col)))

    res[label] = pd.concat(_res).rename_axis(index="comparison")

    res[label].to_csv(output_dir / f"differential.{model_name}.{label}.results.csv")


# # Now try a model of patients only where we regress on time too
model_name = "categoricals+continuous"
cur_variables = categories[:4] + technical + continuous

for m, d, label, fit_vars in [
    (meta, facs, "original", cur_variables),
    (meta_reduced, facs_reduced, "reduced", cur_variables[:-1]),
]:
    # data = zscore(d).join(m[cur_variables]).dropna()
    data = d.join(m[fit_vars]).dropna()

    _res = list()
    for col in d.columns:
        # data[col] = data[col] / 100  # for logit or binomial

        formula = f"{col} ~ {' + '.join(fit_vars)}"
        md = smf.glm(formula, data, family=sm.families.Gamma(sm.families.links.log()))
        mdf = md.fit_regularized(maxiter=100, refit=True)
        params = pd.Series(mdf.params, index=md.exog_names, name="coef")
        pvalues = pd.Series(mdf.pvalues, index=md.exog_names, name="pval")

    res[label] = res[label].append(pd.concat(_res).rename_axis(index="comparison").loc[continuous])

    res[label].to_csv(output_dir / f"differential.{model_name}.{label}.results.csv")


for label in ["original", "reduced"]:

    res[label] = pd.read_csv(output_dir / f"differential.{model_name}.{label}.results.csv")

    long_f = res[label].pivot_table(index="variable", columns="comparison")
    long_f.index = matrix.columns

    changes = long_f["coef"]
    pvals = long_f["pval"]
    logpvals = log_pvalues(pvals).fillna(0)
    qvals = (
        long_f["pval"]
        .apply(multipletests, method="fdr_bh")
        .apply(lambda x: pd.Series(x[1]))
        .T.set_index(long_f.index)
    )
    logqvals = log_pvalues(qvals)

    # Visualize

    # # Heatmaps
    kwargs = dict(center=0, cmap="RdBu_r", robust=True, metric="correlation")
    grid = sns.clustermap(changes, cbar_kws=dict(label="log2(fold-change)"), **kwargs)
    grid.savefig(output_dir / f"differential.{label}.lfc.all_vars.clustermap.svg")
    grid = sns.clustermap(logpvals, cbar_kws=dict(label="-log10(p-value)"), **kwargs)
    grid.savefig(output_dir / f"differential.{label}.pvals_only.all_vars.clustermap.svg")

    # # # Heatmap combinin both change and significance
    cols = ~changes.columns.str.contains("|".join(technical))
    grid = sns.clustermap(
        changes.loc[:, cols].drop("Intercept", axis=1),
        cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
        row_colors=logpvals.loc[:, cols],
        **kwargs,
    )
    grid.savefig(output_dir / f"differential.{label}.join_lfc_pvals.all_vars.clustermap.svg")

    # # # only significatnt
    sigs = (logqvals >= log_alpha_thresh).any(1)
    grid = sns.clustermap(
        changes.loc[sigs, cols].drop("Intercept", axis=1),
        cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
        row_colors=logpvals.loc[sigs, cols],
        xticklabels=True,
        yticklabels=True,
        **kwargs,
    )
    grid.savefig(
        output_dir / f"differential.{label}.join_lfc_pvals.p<{alpha_thresh}_only.clustermap.svg"
    )

    # # Volcano plots
    for category in categories:
        cols = logpvals.columns[logpvals.columns.str.contains(category)]
        n = len(cols)
        if not n:
            continue

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
        axes = axes.squeeze(0)
        for i, col in enumerate(cols):
            sigs = logqvals[col] >= log_alpha_thresh
            kwargs = dict(s=2, alpha=0.5, color="grey")
            axes[i].scatter(changes.loc[~sigs, col], logpvals.loc[~sigs, col], **kwargs)
            kwargs = dict(s=10, alpha=1.0, c=logqvals.loc[sigs, col], cmap="Reds", vmin=0)
            im = axes[i].scatter(changes.loc[sigs, col], logpvals.loc[sigs, col], **kwargs)
            name = re.findall(r"^(.*)\[", col)[0]
            inst = re.findall(r"\[T.(.*)\]", col)[0]
            # v = -np.log10(multipletests([alpha_thresh] * changes[col].shape[0])[1][0])
            v = logpvals.loc[~sigs, col].max()
            axes[i].axhline(v, color="grey", linestyle="--", alpha=0.5)
            axes[i].axvline(0, color="grey", linestyle="--", alpha=0.5)
            axes[i].set_title(f"{name}: {inst}/{meta[category].min()}")
            axes[i].set_xlabel("log2(Fold-change) " + r"($\beta$)")
            axes[i].set_ylabel("-log10(p-value)")

            texts = text(
                changes.loc[sigs, col],
                logpvals.loc[sigs, col],
                changes.loc[sigs, col].index,
                axes[i],
                fontsize=5,
            )
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black"), ax=axes[i])
            add_colorbar(im, axes[i])

        fig.savefig(output_dir / f"differential.{label}.test_{category}.volcano.svg")
        plt.close(fig)

    # # MA plots
    for category in categories:
        cols = logpvals.columns[logpvals.columns.str.contains(category)]
        n = len(cols)
        if not n:
            continue

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
        axes = axes.squeeze(0)
        for i, col in enumerate(cols):
            sigs = logqvals[col] >= log_alpha_thresh
            kwargs = dict(s=2, alpha=0.5, color="grey")
            axes[i].scatter(changes.loc[~sigs, "Intercept"], changes.loc[~sigs, col], **kwargs)
            kwargs = dict(s=10, alpha=1.0, c=logqvals.loc[sigs, col], cmap="Reds", vmin=0)
            im = axes[i].scatter(changes.loc[sigs, "Intercept"], changes.loc[sigs, col], **kwargs)
            name = re.findall(r"^(.*)\[", col)[0]
            inst = re.findall(r"\[T.(.*)\]", col)[0]
            axes[i].axhline(0, color="grey", linestyle="--", alpha=0.5)
            axes[i].set_title(f"{name}: {inst}/{meta[category].min()}")
            axes[i].set_xlabel("Mean")
            axes[i].set_ylabel("log2(fold-change)")

            texts = text(
                changes.loc[sigs, "Intercept"],
                changes.loc[sigs, col],
                changes.loc[sigs, col].index,
                axes[i],
                fontsize=5,
            )
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black"), ax=axes[i])
            add_colorbar(im, axes[i])

        fig.savefig(output_dir / f"differential.{label}.test_{category}.maplots.svg")
        plt.close(fig)

    # # Illustration of top hits
    n_plot = 1000

    for category in categories:
        cols = pvals.columns[pvals.columns.str.contains(category)]
        control = meta[category].min()
        n = len(cols)
        if not n:
            continue
        for i, col in enumerate(cols):
            v = logqvals[col].sort_values()
            sigs = v[v >= log_alpha_thresh]
            if sigs.empty:
                continue
            # print(category, sigs)
            sigs = sigs.tail(n_plot).index[::-1]
            data = matrix.loc[:, sigs].join(meta[category]).melt(id_vars=category)

            kws = dict(data=data, x=category, y="value", hue=category, palette="tab10")
            grid = sns.FacetGrid(data=data, col="variable", sharey=False, height=3, col_wrap=4)
            grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
            # grid.map_dataframe(sns.stripplot, y="value", x=category, hue=category, data=data, palette='tab10')

            for ax in grid.axes.flat:
                [
                    x.set_alpha(0.25)
                    for x in ax.get_children()
                    if isinstance(
                        x,
                        (
                            matplotlib.collections.PatchCollection,
                            matplotlib.collections.PathCollection,
                        ),
                    )
                ]
            grid.map_dataframe(sns.swarmplot, **kws)

            for ax in grid.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

            # add stats to title
            group = re.findall(r"^.*\[T.(.*)\]", col)[0]
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
                    + f"Coef = {changes.loc[var, col]:.3f}; "
                    + f"FDR = {qvals.loc[var, col]:.3e}"
                )

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
