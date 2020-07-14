#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""


import flowkit as fk
from flowsom import flowsom
from anndata import AnnData
import scanpy as sc
from minisom import MiniSom

from imc.operations import (
    get_best_mixture_number,
    get_threshold_from_gaussian_mixture,
)

from src.conf import *


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
    names[0] = names[0].str.replace(" .*", "").replace("Viablity", "Viability")
    names[1] = names.apply(
        lambda x: "(" + x[1] + ")" if x[0] != "" else x[1], 1
    )
    return names[0] + names[1]


def get_population(
    ser: pd.Series, population: int = -1, plot=False, ax=None, **kwargs
) -> pd.Index:
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
    done = False
    while not done:
        try:
            thresh = get_threshold_from_gaussian_mixture(xx, n_components=n)
        except AssertionError:
            continue
        done = True

    sel = xx[operator(xx, thresh.iloc[population])].index

    if plot:
        ax = plt.gca() if ax is None else ax
        sns.distplot(xx, kde=False, ax=ax)
        sns.distplot(xx[sel], kde=False, ax=ax)
        [ax.axvline(q, linestyle="--", color="grey") for q in thresh]
        ax = None
    return sel


output_dir = Path("results") / "single_cell"
output_dir.mkdir(exist_ok=True, parents=True)

panels = json.load(open(metadata_dir / "flow_variables2.json"))
fcs_dir = data_dir / "fcs"

metadata_file = metadata_dir / "annotation.pq"
meta = pd.read_parquet(metadata_file)


plot_gating = False
overwrite = False
n = 2000

pos_gate_names = {"WB_Memory": "CD3+", "WB_IgG_IgM": "CD19+"}
pos_gate_channels = {
    "WB_Memory": "CD3(FITC-A)",
    "WB_IgG_IgM": "CD19(Pacific Blue-A)",
}

# Extract matrix, gate

# panel = "WB_Memory"
# panel = "WB_IgG_IgM"
# sample_id = "S100"
# plot_gating = True

failures = list()

for panel in panels:
    print(panel)
    for sample_id in meta["sample_id"].unique():
        print(sample_id)
        (output_dir / panel).mkdir(exist_ok=True, parents=True)

        sample_name = (
            meta.loc[
                meta["sample_id"] == sample_id, ["patient_code", "sample_id"]
            ]
            .drop_duplicates()
            .squeeze()
            .name
        )
        output_file = output_dir / panel / f"{sample_name}.filtered.csv.gz"
        output_file_subsampled = (
            output_dir / panel / f"{sample_name}.filtered.subsampled.csv.gz"
        )
        output_figure = (
            output_dir / panel / f"{sample_name}.filtering_gating.svg"
        )
        if output_file.exists() and not overwrite:
            continue

        # TODO: check for more files
        _id = int(sample_id.replace("S", ""))
        try:
            fcs_file = sorted(list(fcs_dir.glob(f"{_id}_{panel}*.fcs")))[0]
            # this makes sure files with *(1) are read instead of potentially
            # corrupted ones from Cytobank
        except IndexError:
            try:
                fff = list(fcs_dir.glob(f"{_id}x*_{panel}*.fcs"))
                # assert len(fff) in [0, 1]
                fcs_file = fff[0]
            except IndexError:
                print(f"Sample {sample_id} is missing!")
                failures.append(sample_id)
                continue

        try:
            s = fk.Sample(fcs_file)
            # some corrupted files will fail here but the correct one will be read after
            # e.g. 195_WB_IgG_IgM
        except:
            failures.append(sample_id)
            continue
        ch_names = get_channel_labels(s)
        s.apply_compensation(s.metadata["spill"])

        # xform = fk.transforms.HyperlogTransform("hlog"'hyper', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        # xform = fk.transforms.AsinhTransform("asinh", param_t=10000, param_m=4.5, param_a=0.25)
        xform = fk.transforms.AsinhTransform(
            "asinh", param_t=10000, param_m=4.5, param_a=0
        )
        s.apply_transform(xform)
        # x = pd.DataFrame(s.get_comp_events(), columns=ch_names)
        df = pd.DataFrame(s.get_transformed_events(), columns=ch_names)
        # convert time to seconds
        df["Time"] *= float(s.metadata["timestep"])

        # save
        df.index.name = "cell"
        # df.to_csv(output_dir / panel / f"{sample_name}.csv.gz")
        # df.sample(n=n).to_csv(output_dir / panel / f"{sample_name}.sampled_{n}.csv.gz")

        # # Observe dependency on time
        # t = df['Time']
        # plt.plot(t, t.index)
        # plt.plot([t.min(), t.max()], [t.min(), t.max()])
        # g = df.groupby(pd.cut(t, 100))[df.columns].mean().drop('Time', 1)
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
        if plot_gating:
            fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 4))
            kws = dict(s=2, alpha=0.1, rasterized=True)
            axes[0].scatter(df[x], df[y], c="grey", **kws)
            axes[0].scatter(xdf[x], xdf[y], c=xdf[ratio], cmap="RdBu_r", **kws)
            axes[0].set(xlabel=x, ylabel=y)

        # # 2. Viable
        name = "Viable"
        x = "Viability(APC-R700-A)"
        ax = axes[1] if plot_gating else None
        sel = get_population(xdf[x], 0, plot=plot_gating, ax=ax)
        xdf = xdf.loc[sel]

        # # 3. Population-specific gate
        # name = "CD3+"
        # name = "CD19+"
        name = pos_gate_names[panel]
        # x = "CD3(FITC-A)"
        # x = "CD19(Pacific Blue-A)"
        x = pos_gate_channels[panel]
        ax = axes[2] if plot_gating else None
        sel = get_population(xdf[x], -1, plot=plot_gating, ax=ax)
        xdf = xdf.loc[sel]

        # .iloc[:, 4:-2] <- to remove FSC, Time, etc cols
        xdf.to_csv(output_file)
        try:
            xdf.sample(n=n).to_csv(output_file_subsampled)
        except ValueError:
            xdf.to_csv(output_file_subsampled)

        if plot_gating:
            fig.axes[1].set_yscale("log")
            fig.axes[2].set_yscale("log")
            fig.savefig(output_figure, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"Sample '{sample_id}' has {xdf.shape[0]} filtered cells.")


# Actually run analysis
# panel = "WB_IgG_IgM"

for panel in panels:
    csv_dir = output_dir / panel

    df = pd.concat(
        [
            pd.read_csv(f, index_col=0).assign(sample=f.parts[-1].split(".")[0])
            for f in csv_dir.glob("*.subsampled.csv.gz")
        ]
    )

    # sample 'P060-S074-R01' has channel 'FITC-A' instead of 'sIgG(FITC-A)'
    df.loc[df["sIgG(FITC-A)"].isnull(), "sIgG(FITC-A)"] = df.loc[
        df["sIgG(FITC-A)"].isnull(), "FITC-A"
    ]
    df = df.drop("FITC-A", axis=1).reset_index(drop=True)

    ann = AnnData(df.iloc[:, 4:-3], obs=dict(sample=df["sample"]))


# def further():
#     # drop FSC, SSC, Time
#     x = xdf.iloc[:, 4:-2]

#     # # Select columns to use for selecting cells
#     # cols = x.columns

#     # positives = dict()
#     # n, m = get_grid_dims(len(cols))
#     # fig, axes = plt.subplots(n, m, figsize=(m * 6, n * 4))
#     # axes = axes.flatten()
#     # for i, col in tqdm(enumerate(cols)):
#     #     positives[col] = get_positive_population(df[col], plot=True, ax=axes[i], min_mix=3)
#     # fig.savefig(
#     #     output_dir / f"{sample_name}_{panel}.gaussian_thresholds.min_3.svg",
#     #     bbox_inches="tight",
#     #     dpi=300,
#     # )

#     x2 = x  # .sample(n=10000)
#     print("anndata")
#     a = AnnData(x2.drop(["Viability(APC-R700-A)", "CD3(FITC-A)"], 1))

#     sc.pp.pca(a)
#     sc.pp.neighbors(a)
#     sc.tl.umap(a)
#     sc.tl.leiden(a, key_added="cluster", resolution=0.25)
#     # sc.pl.pca(a, color=a.var.index.tolist() + ["cluster"])
#     # sc.pl.umap(a, color=a.var.index.tolist() + ["cluster"])

#     fig = sc.pl.umap(a, color=a.var.index.tolist() + ["cluster"], show=False)[
#         0
#     ].figure
#     fig.savefig(
#         output_dir / panel / f"{sample_name}.single_cell.umap.all_markers.svg",
#         **figkws,
#     )

#     cluster_means = a.to_df().groupby(a.obs["cluster"]).mean()
#     cluster_means["ratio"] = (
#         cluster_means["CD8(APC-H7-A)"] / cluster_means["CD4(BV605-A)"]
#     )
#     cluster_means = cluster_means.sort_values("ratio")
#     cluster_means = cluster_means.sort_values("CD4(BV605-A)")
#     cells = a.to_df().groupby(a.obs["cluster"]).size().rename("Cells")
#     cells_p = ((cells / cells.sum()) * 100).rename("Cells (%)")
#     cells = pd.concat([cells, cells_p], 1)
#     grid = sns.clustermap(
#         cluster_means.T.drop("ratio"),
#         metric="correlation",
#         z_score=0,
#         cmap="RdBu_r",
#         center=0,
#         robust=True,
#         col_cluster=False,
#         row_colors=cluster_means.mean().rename("Mean"),
#         col_colors=cells,
#         cbar_kws=dict(label="Intensity (Z-score)"),
#         figsize=(6, 4),
#     )
#     grid.savefig(
#         output_dir
#         / panel
#         / f"{sample_name}.single_cell.leiden_clusters.mean.clustermap.svg",
#         **figkws,
#     )

#     for label, sdf in [
#         ("CD4", cluster_means.query("ratio < 1")),
#         ("CD8", cluster_means.query("ratio > 1")),
#     ]:
#         sdf = sdf.loc[:, ~sdf.columns.str.contains(r"CD4\(|CD8\(")]
#         grid = sns.clustermap(
#             sdf.T.drop("ratio"),
#             metric="correlation",
#             # z_score=0,
#             # cmap="RdBu_r",
#             # center=0,
#             # cbar_kws=dict(label="Intensity (Z-score)"),
#             cbar_kws=dict(label="Intensity"),
#             robust=True,
#             col_cluster=True,
#             row_colors=cluster_means.mean().rename("Mean"),
#             col_colors=cells,
#             figsize=(6, 4),
#         )
#         grid.savefig(
#             output_dir
#             / panel
#             / f"{sample_name}.single_cell.leiden_clusters.only_{label}.mean.clustermap.svg",
#             **figkws,
#         )

#     # # MiniSOM
#     # xv = x.values
#     # sx = 6
#     # sy = 6
#     # max_iter = 10000

#     # som = MiniSom(
#     #     sx, sy, xv.shape[1], sigma=0.3, learning_rate=0.5
#     # )  # initialization of 6x6 SOM
#     # som.pca_weights_init(xv)
#     # som.train(xv, max_iter, random_order=True, verbose=True)

#     # w = np.asarray([som.winner(xv[i]) for i in range(xv.shape[0])])
#     # # # get cells from given SOM square
#     # _means = dict()
#     # for i in range(sx):
#     #     for j in range(sy):
#     #         _means[(i, j)] = x[(w == [i, j]).all(1)].mean()
#     # means = pd.DataFrame(_means).rename_axis(columns=["x", "y"])

#     # fig, axes = plt.subplots(1, means.shape[0])
#     # for i, channel in enumerate(sorted(means.index)):
#     #     axes[i].set_title(channel)
#     #     axes[i].imshow(means.T.pivot_table(index="x", columns="y", values=channel))

#     # # update
#     # target = 0.5
#     # error = list()
#     # # for i in range(max_iter):
#     # e = np.inf
#     # i = 0
#     # while e > target:
#     #     if i % 100 == 0:
#     #         print(i)
#     #     som.update(xv[i], som.winner(xv[i]), i, 1)
#     #     e = som.quantization_error(xv)
#     #     error.append(e)
#     #     i += 1

#     # dist = som.distance_map()
#     # som.activation_response(xv)
