#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import flowkit as fk
from flowsom import flowsom
from anndata import AnnData
import scanpy as sc
from minisom import MiniSom


def get_grid_dims(dims, nstart=None):
    """
    Given a number of `dims` subplots, choose optimal x/y dimentions of plotting
    grid maximizing in order to be as square as posible and if not with more
    columns than rows.
    """
    if nstart is None:
        n = min(dims, 1 + int(np.ceil(np.sqrt(dims))))
    else:
        n = nstart
    if (n * n) == dims:
        m = n
    else:
        a = pd.Series(n * np.arange(1, n + 1)) / dims
        m = a[a >= 1].index[0] + 1
    assert n * m >= dims

    if n * m % dims > 1:
        try:
            n, m = get_grid_dims(dims=dims, nstart=n - 1)
        except IndexError:
            pass
    return n, m


def get_channel_labels(sample):
    names = pd.DataFrame([s.pns_labels, s.pnn_labels]).T
    names[0] = names[0].str.replace(" .*", "")
    return names[0] + "(" + names[1] + ")"


def get_positive_population(x, plot=False, ax=None, **kwargs):
    # from imc.operations import get_best_mixture_number, get_threshold_from_gaussian_mixture

    xx = x[x > 0]
    n, y = get_best_mixture_number(xx, return_prediction=True, **kwargs)
    thresh = get_threshold_from_gaussian_mixture(xx, n_components=n)
    if plot:
        if ax is None:
            ax = plt.gca()
        sns.distplot(xx, kde=False, ax=ax)
        sns.distplot(xx[xx >= thresh.iloc[-1]], kde=False, ax=ax)
        [ax.axvline(q, linestyle="--", color="grey") for q in thresh]
    return xx[xx >= thresh.iloc[-1]]


data_dir = Path("data")
output_dir = Path("results") / "single_cell"
output_dir.mkdir(exist_ok=True, parents=True)

panels = {
    # "Treg": {},
    # "T3": {}
    #
    "Checkpoint": {
        "id": "WB_Checkpoint",
        "num": "309657",
        "channels": [
            "Viablity(APC-R700-A)",
            "CD3(FITC-A)",
            "CD4(BV605-A)",
            "CD8(APC-H7-A)",
            "CD152(PE-A)",  # CTLA4
            "TIGIT(PerCP-Cy5-5-A)",
            "CD366(PE-Cy7-A)",  # Tim3
            "VISTA(APC-A)",
            "CD279(Pacific Blue-A)",  # PD-1
            "CD223(V500C-A)",  # Lag3
        ],
    },
    "Ig": {
        "id": "WB_IgG_IgM",
        "num": "309658",
        "channels": [
            "sIgG(FITC-A)",
            "CD25(PE-A)",
            "CD27(PerCP-Cy5-5-A)",
            "CD10(PE-Cy7-A)",
            "sIgM(APC-A)",
            "Viability(APC-R700-A)",
            "CD20(APC-H7-A)",
            "CD19(Pacific Blue-A)",
            "CD45(V500C-A)",
            "CD5(BV605-A)",
        ],
    },
    "PBMC": {
        "id": "PBMC_MDSC",
        "num": "309659",
        "channels": [
            "CD15(FITC-A)",
            "CD33(PE-A)",
            "HLA-DR(PerCP-Cy5-5-A)",
            "CD3(PE-Cy7-A)",
            "CD11b(APC-A)",
            "Viability(APC-R700-A)",
            "CD14(APC-H7-A)",
            "CD124(Pacific Blue-A)",
            "CD45(V500C-A)",
            "CD16(BV605-A)",
        ],
    },
    "Memory": {
        "id": "WB_Memory",
        "num": "309660",
        "channels": [
            "Viability(APC-R700-A)",
            "CD3(FITC-A)",
            "CD4(BV605-A)",
            "CD8(APC-H7-A)",
            "CD45RA(PE-Cy7-A)",
            "CD45RO(PerCP-Cy5-5-A)",
            "CD25(PE-A)",  # POS: Tregs; high CD45RO, low CD45RA
            "CD197(APC-A)",  # CCR7
            "CD95(Pacific Blue-A)",  # Fas
            "CD62L(V500C-A)",  # High: Naive, Low: Memory; Memory # POS: Central memory, NEG: effector memory
        ],
        # split CD4 - CD8
        # split CD45RA - CD45RO
        # split in combs of CD197, CD62L, CD95 (-/+)
        # # CD197-CD62L-, CD197+CD62L-, CD197-CD62L+, CD197+CD62L+
    },
}

sample_name = "102"

names = dict()
for panel in panels:
    print(panel)
    fcs_dir = data_dir / "fcs" / panels[panel]["num"]
    # TODO: check for more files
    fcs_file = list(fcs_dir.glob(sample_name + "_" + panels[panel]["id"] + "*.fcs"))[0]

    s = fk.Sample(fcs_file)
    ch_names = get_channel_labels(s)

    names[panel] = ch_names
    s.apply_compensation(s.metadata["spill"])

    # xform = fk.transforms.HyperlogTransform("hlog"'hyper', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
    xform = fk.transforms.AsinhTransform("asinh", param_t=10000, param_m=4.5, param_a=0)
    s.apply_transform(xform)
    # x = pd.DataFrame(s.get_comp_events(), columns=ch_names)
    x = pd.DataFrame(s.get_transformed_events(), columns=ch_names)
    # convert time to seconds
    x["(Time)"] *= float(s.metadata["timestep"])

    # drop FSC, SSC, Time
    x = x.loc[:, ~x.columns.str.startswith("(")]

    # Select columns to use for selecting cells
    cols = x.columns

    positives = dict()
    n, m = get_grid_dims(len(cols))
    fig, axes = plt.subplots(n, m, figsize=(m * 6, n * 4))
    axes = axes.flatten()
    for i, col in tqdm(enumerate(cols)):
        positives[col] = get_positive_population(x[col], plot=True, ax=axes[i], min_mix=3)
    fig.savefig(
        output_dir / f"{sample_name}_{panel}.gaussian_thresholds.min_3.svg",
        bbox_inches="tight",
        dpi=300,
    )

    x2 = x.sample(n=10000)
    print("anndata")
    a = AnnData(x2)

    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    sc.pl.pca(a, color=a.var.index)
    # sc.pl.umap(a, color=a.var.index)

    xv = x.values
    sx = 6
    sy = 6
    max_iter = 10000

    som = MiniSom(sx, sy, xv.shape[1], sigma=0.3, learning_rate=0.5)  # initialization of 6x6 SOM
    som.pca_weights_init(xv)
    som.train(xv, max_iter, random_order=True, verbose=True)

    w = np.asarray([som.winner(xv[i]) for i in range(xv.shape[0])])
    # # get cells from given SOM square
    _means = dict()
    for i in range(sx):
        for j in range(sy):
            _means[(i, j)] = x[(w == [i, j]).all(1)].mean()
    means = pd.DataFrame(_means).rename_axis(columns=["x", "y"])

    fig, axes = plt.subplots(1, means.shape[0])
    for i, channel in enumerate(sorted(means.index)):
        axes[i].set_title(channel)
        axes[i].imshow(means.T.pivot_table(index="x", columns="y", values=channel))

    # # update
    # target = 0.5
    # error = list()
    # # for i in range(max_iter):
    # e = np.inf
    # i = 0
    # while e > target:
    #     if i % 100 == 0:
    #         print(i)
    #     som.update(xv[i], som.winner(xv[i]), i, 1)
    #     e = som.quantization_error(xv)
    #     error.append(e)
    #     i += 1

    # dist = som.distance_map()
    # som.activation_response(xv)
