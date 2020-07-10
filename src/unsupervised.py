#!/usr/bin/env python

"""
"""

import json
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS, TSNE
from umap import UMAP

import imc  # this import is here to allow automatic colorbars in clustermap
from imc.graphics import to_color_series

from src.conf import *


def fix_clustermap_fonts(grid, fontsize=3):
    grid.ax_heatmap.set_yticklabels(
        grid.ax_heatmap.get_yticklabels(), fontsize=fontsize, ha="left"
    )
    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xticklabels(), fontsize=fontsize, va="top"
    )
    grid.ax_row_colors.set_yticklabels(
        grid.ax_row_colors.get_yticklabels(), fontsize=fontsize, ha="left"
    )
    grid.ax_row_colors.set_xticklabels(
        grid.ax_row_colors.get_xticklabels(), fontsize=fontsize, va="top"
    )


def plot_projection(x, meta, cols, n_dims=4, algo_name="PCA"):
    n = len(cols)
    fig, axes = plt.subplots(
        n,
        n_dims,
        figsize=(4 * n_dims, 4 * n),
        sharex="col",
        sharey="col",
        squeeze=False,
    )

    for i, cat in enumerate(cols):
        colors = to_color_series(meta[cat])
        for pc in x.columns[:n_dims]:
            for value in meta[cat].unique():
                idx = meta[cat].isin([value])  # to handle nan correctly
                m = axes[i, pc].scatter(
                    x.loc[idx, pc],
                    x.loc[idx, pc + 1],
                    c=colors.loc[idx],
                    label=value,
                )
            if pc == 0:
                axes[i, pc].legend(
                    title=cat, loc="center right", bbox_to_anchor=(-0.15, 0.5)
                )
            axes[i, pc].set_ylabel(algo_name + str(pc + 2))

    for i, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(algo_name + str(i + 1))
    return fig


output_dir = results_dir / "unsupervised"

meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file)
matrix_reduced = pd.read_parquet(matrix_imputed_reduced_file)


categories = CATEGORIES
continuous = CONTINUOUS
sample_variables = meta[categories + continuous]

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
    .join(
        matrix.loc[meta["patient"] == "Control"].mean().rename("Mean control")
    )
    .join(
        matrix.loc[meta["patient"] == "Patient"].mean().rename("Mean patient")
    )
)
# Demonstrate the data


# # Plot abundance of major populations for each patient group
# + a few ratios like CD4/CD8 (of CD3+)
for cat_var in categories:
    # cat_var = "severity_group"

    for panel in variable_classes["panel"].unique():
        figfile = (
            output_dir
            / f"variable_illustration.{cat_var}.panel_{panel}.swarm+boxen.svg"
        )
        # if figfile.exists():
        #     continue

        data = (
            matrix.loc[:, variable_classes.query(f"panel == '{panel}'").index]
            .join(meta[[cat_var]])
            .melt(
                id_vars=[cat_var],
                var_name="population",
                value_name="abundance (%)",
            )
        )

        kws = dict(
            data=data,
            x=cat_var,
            y="abundance (%)",
            hue=cat_var,
            palette="tab10",
        )
        grid = sns.FacetGrid(
            data=data, col="population", sharey=False, height=3, col_wrap=4
        )
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
        for ax in grid.axes.flat:
            var = ax.get_title().replace("population = ", "")
            try:
                child, parent = re.findall(r"(.*)/(.*)", var)[0]
                ax.set_title(child)
                ax.set_ylabel(f"% {parent}")
            except IndexError:
                ax.set_title(var)

        # grid.map(sns.boxplot)
        grid.savefig(figfile)
        plt.close(grid.fig)


# # Simply correlate with clinical continuous
for num_var in continuous:
    for panel in variable_classes["panel"].unique():
        figfile = (
            output_dir
            / f"variable_illustration.{num_var}.panel_{panel}.swarm+boxen.svg"
        )
        # if figfile.exists():
        #     continue

        data = (
            matrix.loc[:, variable_classes.query(f"panel == '{panel}'").index]
            .join(meta[[num_var]])
            .melt(
                id_vars=[num_var],
                var_name="population",
                value_name="abundance (%)",
            )
        )

        kws = dict(data=data, x=num_var, y="abundance (%)")
        grid = sns.FacetGrid(
            data=data, col="population", sharey=False, height=3, col_wrap=4
        )
        grid.map_dataframe(sns.regplot, **kws)

        # add stats to title
        for ax in grid.axes.flat:
            var = ax.get_title().replace("population = ", "")
            try:
                child, parent = re.findall(r"(.*)/(.*)", var)[0]
                ax.set_title(child)
                ax.set_ylabel(f"% {parent}")
            except IndexError:
                ax.set_title(var)
            ax.set_xlabel(num_var)

        # grid.map(sns.boxplot)
        grid.savefig(figfile)
        plt.close(grid.fig)

# Clustermaps

# # all samples, all variables, full or reduced

for df, label1 in [(matrix, "full"), (matrix_reduced, "reduced")]:
    prefix = f"covid-facs.cell_type_abundances.{label1}."
    kwargs = dict(
        metric="correlation",
        robust=True,
        figsize=(12, 8),
        row_colors=sample_variables,
        col_colors=variable_classes.loc[df.columns],
        colors_ratio=(
            0.15 / sample_variables.shape[1],
            0.15 / variable_classes.loc[df.columns].shape[1],
        ),
        dendrogram_ratio=0.1,
        rasterized=True,
        xticklabels=True,
        yticklabels=True,
    )
    # # # original values
    grid = sns.clustermap(
        df,
        cbar_kws=dict(
            label="Cell type abundance (%)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
        ),
        **kwargs,
    )
    fix_clustermap_fonts(grid)
    grid.savefig(output_dir / (prefix + "clustermap.percentage.svg"), **figkws)
    plt.close(grid.fig)

    # # # zscore
    grid = sns.clustermap(
        df,
        z_score=1,
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(
            label="Cell type abundance\n(Z-score)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
        ),
        **kwargs,
    )
    fix_clustermap_fonts(grid)
    grid.savefig(output_dir / (prefix + "clustermap.zscore.svg"), **figkws)
    plt.close(grid.fig)

    # # sample correlation
    # # variable correlation

    for df2, label2, colors in [
        (df, "variable", variable_classes.loc[df.columns]),
        (df.T, "sample", sample_variables),
    ]:
        kws = kwargs.copy()
        kws.update(
            dict(
                figsize=(8, 8),
                center=0,
                row_colors=colors,
                col_colors=colors,
                colors_ratio=(0.15 / colors.shape[1], 0.15 / colors.shape[1]),
            )
        )
        grid = sns.clustermap(
            df2.corr(),
            cbar_kws=dict(
                label=f"{label2} correlation",  # , orientation="horizontal", aspect=0.2, shrink=0.2
            ),
            **kws,
        )
        fix_clustermap_fonts(grid)
        grid.savefig(
            output_dir / (prefix + f"{label2}_correlation.clustermap.svg"),
            **figkws,
        )
        plt.close(grid.fig)

    # # Do the same for the major components, LY, CD3, CD20, Myeloid, etc...
    # # or for each parent

    for panel in variable_classes.loc[df.columns]["panel"].unique():
        q = variable_classes.loc[df.columns]["panel"] == panel
        if df.loc[:, q].shape[1] < 2:
            continue

        # kws = kwargs.copy()
        # kws.update(dict(figsize=np.asarray(df.loc[:, q].shape) * 0.05))
        grid = sns.clustermap(
            df.loc[:, q],
            z_score=1,
            cmap="RdBu_r",
            center=0,
            cbar_kws=dict(
                label="Cell type abundance\n(Z-score)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
            ),
            **kwargs,
        )
        fix_clustermap_fonts(grid)
        grid.savefig(
            output_dir / (prefix + f"only_{panel}.clustermap.svg"), **figkws
        )
        plt.close(grid.fig)

    for population in parent_population.unique():
        q = parent_population == population
        if sum(q) < 2:
            continue
        # kws = kwargs.copy()
        # kws.update(dict(figsize=np.asarray(df.loc[:, q].shape) * 0.05))
        grid = sns.clustermap(
            df.loc[:, q],
            z_score=1,
            cmap="RdBu_r",
            center=0,
            cbar_kws=dict(
                label="Cell type abundance\n(Z-score)",  # , orientation="horizontal", aspect=0.2, shrink=0.2
            ),
            **kwargs,
        )
        fix_clustermap_fonts(grid)
        grid.savefig(
            output_dir / (prefix + f"only_{population}.clustermap.svg"),
            **figkws,
        )
        plt.close(grid.fig)


# highly variable variables

# # variant stabilization

# # clustermaps

# # manifolds


# manifold learning

# manifolds = dict()

for model, kwargs in [
    (PCA, dict()),
    (NMF, dict()),
    (MDS, dict(n_dims=1)),
    (TSNE, dict(n_dims=1)),
    (UMAP, dict(n_dims=1)),
][::-1]:
    name = str(model).split(".")[-1].split("'")[0]
    model_inst = model()

    # manifolds[name] = dict()
    for df, label in [(matrix, "percentages"), (zscore(matrix), "zscore")]:
        try:  #  this will occur for example in NMF with Z-score transform
            res = pd.DataFrame(model_inst.fit_transform(df), index=df.index)
        except ValueError:
            continue

        fig = plot_projection(
            res, meta, cols=sample_variables.columns, algo_name=name, **kwargs
        )
        fig.savefig(
            output_dir / f"covid-facs.cell_type_abundances.{name}.{label}.svg",
            **figkws,
        )
        plt.close(fig)

        # manifolds[name][label] = res


# Add lock file
open(output_dir / "__done__", "w")
