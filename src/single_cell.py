#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

from pathlib import Path
import json

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

from imc.operations import get_best_mixture_number, get_threshold_from_gaussian_mixture


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
    names[1] = names.apply(lambda x: "(" + x[1] + ")" if x[0] != "" else x[1], 1)
    return names[0] + names[1]


def get_population(ser: pd.Series, population: int = -1, plot=False, ax=None, **kwargs) -> pd.Index:
    # from imc.operations import get_best_mixture_number, get_threshold_from_gaussian_mixture

    # xx = s[s > 0]
    if population == -1:
        operator = np.greater_equal
    elif population == 0:
        operator = np.less_equal
    else:
        raise ValueError("")

    xx = ser + abs(ser.min())
    n, y = get_best_mixture_number(xx, return_prediction=True, **kwargs)
    thresh = get_threshold_from_gaussian_mixture(xx, n_components=n)

    sel = xx[operator(xx, thresh.iloc[population])].index

    if plot:
        ax = plt.gca() if ax is None else ax
        sns.distplot(ser, kde=False, ax=ax)
        sns.distplot(ser[sel], kde=False, ax=ax)
        [ax.axvline(q, linestyle="--", color="grey") for q in thresh]
        ax = None
    return sel


data_dir = Path("data")
metadata_dir = Path("metadata")
output_dir = Path("results") / "single_cell"
output_dir.mkdir(exist_ok=True, parents=True)

panels = json.load(open(metadata_dir / "flow_variables.json"))
metadata_file = metadata_dir / "annotation.pq"
meta = pd.read_parquet(metadata_file)


n = 2000

# Extract matrix, gate
for panel in panels:
    print(panel)
    for sample_id in meta["sample_id"].unique():
        print(sample_id)
        (output_dir / panel).mkdir(exist_ok=True, parents=True)

        sample_name = (
            meta.loc[meta["sample_id"] == sample_id, ["patient_code", "sample_id"]]
            .drop_duplicates()
            .squeeze()
            .name
        )

        fcs_dir = data_dir / "fcs" / panels[panel]["num"]
        # TODO: check for more files
        _id = int(sample_id.replace("S", ""))
        try:
            fcs_file = list(fcs_dir.glob(f"{_id}_" + panels[panel]["id"] + "*.fcs"))[0]
        except IndexError:
            try:
                fff = list(fcs_dir.glob(f"{_id}x" + "*.fcs"))
                # assert len(fff) in [0, 1]
                fcs_file = fff[0]
            except IndexError:
                print(f"Sample {sample_id} is missing!")
                continue

        s = fk.Sample(fcs_file)
        ch_names = get_channel_labels(s)
        s.apply_compensation(s.metadata["spill"])

        # xform = fk.transforms.HyperlogTransform("hlog"'hyper', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        xform = fk.transforms.AsinhTransform("asinh", param_t=10000, param_m=4.5, param_a=0)
        s.apply_transform(xform)
        # x = pd.DataFrame(s.get_comp_events(), columns=ch_names)
        df = pd.DataFrame(s.get_transformed_events(), columns=ch_names)
        # convert time to seconds
        df["Time"] *= float(s.metadata["timestep"])

        # save
        df.index.name = "cell"
        df.to_csv(output_dir / panel / f"{sample_name}.csv.gz")
        df.sample(n=n).to_csv(output_dir / panel / f"{sample_name}.sampled_{n}.csv.gz")

        # # Observe dependency on time
        # t = x['(Time)']
        # plt.plot(t, t.index)
        # plt.plot([t.min(), t.max()], [t.min(), t.max()])
        # g = x.groupby(pd.cut(t, 100))[x.columns].mean().drop('(Time)', 1)
        # _, axes = plt.subplots(len(g.columns))
        # [ax.plot(g[z]) for ax, z in zip(axes, g.columns)]

        # Gate
        # # 1. Single cells
        name = "singlets"
        x = "FSC-H"
        y = "FSC-A"
        ratio = "FSC-H:FSC-A_ratio"
        min_x = 50_000
        max_x = 225_000

        min_y = 80_000
        max_y = 225_000
        max_ratio = 2
        df[ratio] = df[y] / df[x]

        xdf = df.loc[
            (df[x] > min_x)
            & (df[x] < max_x)
            & (df[y] > min_y)
            & (df[y] < max_y)
            & (df[ratio] < max_ratio)
        ]
        plt.scatter(df[x], df[y], s=2, alpha=0.1, c="grey")
        plt.scatter(xdf[x], xdf[y], s=2, alpha=0.1, c=xdf[ratio], cmap="RdBu_r")

        # # 2. Viable
        name = "Viable"
        x = "Viability(APC-R700-A)"
        sel = get_population(xdf[x], 0)
        xdf = xdf.loc[sel]

        # # 3. CD3+
        name = "CD3+"
        x = "CD3(FITC-A)"
        sel = get_population(xdf[x], -1, plot=True)
        xdf = xdf.loc[sel]

        # drop FSC, SSC, Time
        x = df.loc[:, ~x.columns.str.startswith("(")]

        # Select columns to use for selecting cells
        cols = x.columns

        positives = dict()
        n, m = get_grid_dims(len(cols))
        fig, axes = plt.subplots(n, m, figsize=(m * 6, n * 4))
        axes = axes.flatten()
        for i, col in tqdm(enumerate(cols)):
            positives[col] = get_positive_population(df[col], plot=True, ax=axes[i], min_mix=3)
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

        som = MiniSom(
            sx, sy, xv.shape[1], sigma=0.3, learning_rate=0.5
        )  # initialization of 6x6 SOM
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
