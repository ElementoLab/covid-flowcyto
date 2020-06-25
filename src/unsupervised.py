#!/usr/bin/env python

"""
"""

from imc.types import Path
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from fancyimpute import MatrixFactorization
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS, TSNE
from umap import UMAP

import imc
from imc.graphics import to_color_series


def plot_projection(x, meta, cols, n_dims=4, algo_name="PCA"):
    n = len(cols)
    fig, axes = plt.subplots(
        n, n_dims, figsize=(4 * n_dims, 4 * n), sharex="col", sharey="col", squeeze=False
    )

    for i, cat in enumerate(cols):
        colors = to_color_series(meta[cat])
        for pc in x.columns[:n_dims]:
            for value in meta[cat].unique():
                idx = meta[cat] == value
                m = axes[i, pc].scatter(
                    x.loc[idx, pc], x.loc[idx, pc + 1], c=colors.loc[idx], label=value
                )
            if pc == 0:
                axes[i, pc].legend(title=cat, loc="center right", bbox_to_anchor=(-0.15, 0.5))
            axes[i, pc].set_ylabel(algo_name + str(pc + 2))

    for i, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(algo_name + str(i + 1))
    return fig


def zscore(x, axis=0):
    return (x - x.mean(axis)) / x.std(axis)


figkws = dict(dpi=300, bbox_inches="tight")

original_dir = Path("data") / "original"
metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results")
output_dir = results_dir / "unsupervised"
output_dir.mkdir(exist_ok=True, parents=True)

for _dir in [original_dir, metadata_dir, data_dir, results_dir]:
    _dir.mkdir(exist_ok=True, parents=True)

metadata_file = metadata_dir / "annotation.pq"
matrix_file = data_dir / "matrix.pq"
matrix_imputed_file = data_dir / "matrix_imputed.pq"

meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file)


categories = ["patient", "severity_group", "intubated", "death", "heme", "bmt", "obesity"]
technical = ["date"]
continuous = ["timepoint"]
sample_variables = meta[categories + continuous + technical]

cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns
parent_population = cols[1].rename("parent_population")

panel_variables = json.load(open(metadata_dir / "panel_variables.json"))
panel_variables = {x: k for k, v in panel_variables.items() for x in v}
panel = {col: panel_variables[col] for col in matrix.columns}

variable_classes = (
    parent_population.to_frame()
    .join(pd.Series(panel, name="panel"))
    .join(matrix.mean().rename("Mean"))
    .join(matrix.loc[meta["patient"] == "Control"].mean().rename("Mean control"))
    .join(matrix.loc[meta["patient"] == "Patient"].mean().rename("Mean patient"))
)
# Demonstrate the data


# # Plot abundance of major populations for each patient group
# + a few ratios like CD4/CD8 (of CD3+)


# # Simply correlate with clinical continuous


prefix = "covid-facs.cell_type_abundances."
# Clustermaps

# # all samples, all variables
grid = sns.clustermap(
    matrix,
    metric="correlation",
    robust=True,
    figsize=(12, 8),
    cbar_kws=dict(
        label="Cell type abundance (%)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
    ),
    row_colors=sample_variables,
    col_colors=variable_classes,
    rasterized=True,
    xticklabels=True,
    yticklabels=True,
)
grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=3)
grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=4)
grid.savefig(output_dir / prefix + "clustermap.percentage.svg", **figkws)

grid = sns.clustermap(
    matrix,
    z_score=1,
    metric="correlation",
    cmap="RdBu_r",
    center=0,
    robust=True,
    figsize=(12, 8),
    cbar_kws=dict(
        label="Cell type abundance\n(Z-score)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
    ),
    row_colors=sample_variables,
    col_colors=variable_classes,
    rasterized=True,
    xticklabels=True,
    yticklabels=True,
)
grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=3)
grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=4)
grid.savefig(output_dir / prefix + "clustermap.zscore.svg", **figkws)


# # sample correlation
# # variable correlation

for df, label, colors in [
    (matrix, "variable", variable_classes),
    (matrix.T, "sample", sample_variables),
]:
    grid = sns.clustermap(
        df.corr(),
        metric="correlation",
        cmap="RdBu_r",
        cbar_kws=dict(
            label=f"{label} correlation",  # , orientation="horizontal", aspect=0.2, shrink=0.2
        ),
        row_colors=colors,
        col_colors=colors,
        rasterized=True,
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=3)
    grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), fontsize=4)
    grid.savefig(output_dir / prefix + f"{label}_correlation.clustermap.svg", **figkws)
    plt.close(grid.fig)


# # Do the same for the major components, LY, CD3, CD20, Myeloid, etc...
# # or for each parent


for panel in variable_classes["panel"].unique():
    q = variable_classes["panel"] == panel
    if matrix.loc[:, q].shape[1] < 2:
        continue
    grid = sns.clustermap(
        matrix.loc[:, q],
        z_score=1,
        center=0,
        robust=True,
        metric="correlation",
        cmap="RdBu_r",
        cbar_kws=dict(
            label="Cell type abundance\n(Z-score)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
        ),
        row_colors=sample_variables,
        col_colors=variable_classes.loc[q],
        rasterized=True,
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=3)
    grid.savefig(output_dir / prefix + f"only_{panel}.clustermap.svg", **figkws)
    plt.close(grid.fig)


for population in parent_population.unique():
    q = parent_population == population
    if sum(q) < 2:
        continue
    grid = sns.clustermap(
        matrix.loc[:, q],
        z_score=1,
        center=0,
        robust=True,
        metric="correlation",
        cmap="RdBu_r",
        cbar_kws=dict(
            label="Cell type abundance\n(Z-score)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
        ),
        row_colors=sample_variables,
        col_colors=variable_classes.loc[q],
        rasterized=True,
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), fontsize=3)
    grid.savefig(output_dir / prefix + f"only_{population}.clustermap.svg", **figkws)
    plt.close(grid.fig)


# highly variable variables

# # variant stabilization

# # clustermaps

# # manifolds


# manifold learning

manifolds = dict()

for model, kwargs in [
    (PCA, dict()),
    (NMF, dict()),
    (MDS, dict(n_dims=1)),
    (TSNE, dict(n_dims=1)),
    (UMAP, dict(n_dims=1)),
]:
    name = str(model).split(".")[-1].split("'")[0]
    model_inst = model()

    manifolds[name] = dict()
    for df, label in [(matrix, "percentages"), (zscore(matrix), "zscore")]:
        try:  #  this will occur for example in NMF with Z-score transform
            res = pd.DataFrame(model_inst.fit_transform(df), index=df.index)
        except ValueError:
            continue

        fig = plot_projection(res, meta, cols=sample_variables.columns, algo_name=name, **kwargs)
        fig.savefig(output_dir / f"covid-facs.cell_type_abundances.{name}.{label}.svg", **figkws)
        plt.close(fig)

        manifolds[name][label] = res


# Add lock file
open(output_dir / "__done__", "w")
