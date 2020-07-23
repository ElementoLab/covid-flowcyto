#!/usr/bin/env python

"""
Prepare single cell data from FCS files in to H5ad format.
"""

import re

import flowkit as fk
from anndata import AnnData
import scanpy as sc

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
    done = False
    while not done:
        try:
            n, y = get_best_mixture_number(xx, return_prediction=True, **kwargs)
        except ValueError:  # "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)"
            continue
        done = True
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

pos_gate_names = {
    "WB_Memory": "CD3+",
    "WB_IgG_IgM": "CD19+",
    "WB_Checkpoint": "CD3+",
    "WB_Treg": "CD3+",
}
pos_gate_channels = {
    "WB_Memory": "CD3(FITC-A)",
    "WB_IgG_IgM": "CD19(Pacific Blue-A)",
    "WB_Checkpoint": "CD3(FITC-A)",
    "WB_Treg": "sCD3(FITC-A)",
}

# Extract matrix, gate

# panel = "WB_Memory"
# panel = "WB_IgG_IgM"
# sample_id = "S100"
# plot_gating = True

failures = list()

# for panel in panels:
for panel in pos_gate_names:
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
                failures.append((panel, sample_id))
                continue

        try:
            s = fk.Sample(fcs_file)
            # some corrupted files will fail here but the correct one will be read after
            # e.g. 195_WB_IgG_IgM
        except:
            failures.append((panel, sample_id))
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


# Concatenate

meta_m = meta.copy().drop(["other", "flow_comment"], axis=1)

# # since h5ad cannot serialize datetime, let's convert to str
for col in meta_m.columns[meta_m.dtypes == "datetime64[ns]"]:
    meta_m[col] = meta_m[col].astype(str)

for panel in pos_gate_names:
    panel_dir = output_dir / panel

    for label, func in [("full", np.logical_not), ("subsampled", np.identity)]:
        df = pd.concat(
            [
                pd.read_csv(f, index_col=0).assign(
                    sample=f.parts[-1].split(".")[0]
                )
                for f in panel_dir.glob("*.csv.gz")
                if func(int("subsampled" in str(f))).squeeze()
            ]
        )
        cell = df.index
        df = df.reset_index(drop=True)
        if panel == "WB_IgG_IgM":
            # sample 'P060-S074-R01' has channel 'FITC-A' instead of 'sIgG(FITC-A)'
            df.loc[df["sIgG(FITC-A)"].isnull(), "sIgG(FITC-A)"] = df.loc[
                df["sIgG(FITC-A)"].isnull(), "FITC-A"
            ]
            df = df.drop("FITC-A", axis=1)

            # some samples have channel 'PE-A' instead of 'CD25(PE-A)'
            df.loc[df["CD25(PE-A)"].isnull(), "CD25(PE-A)"] = df.loc[
                df["CD25(PE-A)"].isnull(), "PE-A"
            ]
            df = df.drop("PE-A", axis=1)

            # some samples have channel 'PerCP-Cy5-5-A' instead of 'CD27(PerCP-Cy5-5-A)'
            df.loc[
                df["CD27(PerCP-Cy5-5-A)"].isnull(), "CD27(PerCP-Cy5-5-A)"
            ] = df.loc[df["CD27(PerCP-Cy5-5-A)"].isnull(), "PerCP-Cy5-5-A"]
            df = df.drop("PerCP-Cy5-5-A", axis=1)
        if panel == "WB_Treg":
            # Channel BV605-A is empty
            df = df.drop("BV605-A", axis=1)

        # x = df.iloc[:, 4:-3]
        x = df.loc[:, ~df.columns.str.contains("FSC|SSC|_ratio|Time|sample")]

        # merge with metadata
        df = df.drop(x.columns, axis=1).merge(
            meta_m, left_on="sample", right_on="sample_name"
        )

        a = AnnData(x, obs=df)
        a.write_h5ad(panel_dir / f"{panel}.concatenated.{label}.h5ad")
