#!/usr/bin/env python

"""
Prepare single cell data from FCS files in to H5ad format.
"""

import struct

import flowkit as fk  # type: ignore
from anndata import AnnData  # type: ignore

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


def gate_dataframe(
    data,
    gating_strategy,
    start_with_singlet_gate: bool = True,
    plot: bool = False,
    output_img: Path = None,
):
    n_gates = len(gating_strategy)
    if start_with_singlet_gate:
        n_gates += 1
        # # 1. Single cells, this is fixed for all
        x = "FSC-H"
        y = "FSC-A"
        ratio = "FSC-H:FSC-A_ratio"
        min_x = 50_000
        max_x = 225_000
        min_y = 80_000
        max_y = 225_000
        max_ratio = 2
        data[ratio] = data[y] / data[x]
        xdata = data.loc[
            (data[x] > min_x)
            & (data[x] < max_x)
            & (data[y] > min_y)
            & (data[y] < max_y)
            & (data[ratio] < max_ratio)
        ]
    else:
        xdata = data
    if plot:
        fig, axes = plt.subplots(1, n_gates, figsize=(n_gates * 4, 4))
        axes = iter(axes)
        if start_with_singlet_gate:
            ax = next(axes)
            kws = dict(s=2, alpha=0.1, rasterized=True)
            ax.scatter(data[x], data[y], c="grey", **kws)
            ax.scatter(xdata[x], xdata[y], c=xdata[ratio], cmap="RdBu_r", **kws)
            ax.set(xlabel=x, ylabel=y)

    # # 2+. Population-specific gates (includes viability)
    for channel, population in gating_strategy:
        sel = get_population(
            xdata[channel],
            population,
            plot=plot,
            ax=next(axes) if plot else None,
        )
        xdata = xdata.loc[sel]

    if plot:
        fig.axes[1].set_yscale("log")
        fig.axes[2].set_yscale("log")
        fig.savefig(output_img, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return xdata


output_dir = Path("results") / "single_cell"
output_dir.mkdir(exist_ok=True, parents=True)

panels = json.load(open(metadata_dir / "flow_variables2.json"))
fcs_dir = data_dir / "fcs"

metadata_file = metadata_dir / "annotation.pq"
meta = pd.read_parquet(metadata_file)


plot_gates = False
overwrite = False
n = 2000

# Extract matrix, gate
failures = list()

# for panel in panels:
for panel_name in gating_strategies:
    print(panel_name)
    for sample_id in meta["sample_id"].unique():
        print(sample_id)
        (output_dir / panel_name).mkdir(exist_ok=True, parents=True)

        sample_name = (
            meta.query(f"sample_id == '{sample_id}'")[
                ["patient_code", "sample_id"]
            ]
            .drop_duplicates()
            .squeeze()
            .name
        )
        prefix = output_dir / panel_name / f"{sample_name}"
        output_file = prefix + ".filtered.csv.gz"
        output_file_subsampled = prefix + ".filtered.subsampled.csv.gz"
        output_figure = prefix + ".filtering_gating.svg"
        if output_file.exists() and not overwrite:
            continue

        _id = int(sample_id.replace("S", ""))
        try:
            fcs_file = sorted(list(fcs_dir.glob(f"{_id}_{panel_name}*.fcs")))[0]
            # this makes sure files with *(1) are read instead of potentially
            # corrupted ones from Cytobank
        except IndexError:
            try:
                fff = list(fcs_dir.glob(f"{_id}x*_{panel_name}*.fcs"))
                fcs_file = fff[0]
            except IndexError:
                print(f"Sample '{sample_id}' is missing!")
                failures.append((panel_name, sample_id))
                continue

        try:
            s = fk.Sample(fcs_file)
            # some corrupted files may fail here but the correct one will be read after
            # e.g. 195_WB_IgG_IgM
            # EDIT: with sorting files above, this should no longer be an issue
        except struct.error:
            print(f"Failed to open file for sample '{sample_id}!")
            failures.append((panel_name, sample_id))
            continue
        ch_names = get_channel_labels(s)
        s.apply_compensation(s.metadata["spill"])

        # xform = fk.transforms.HyperlogTransform("hlog"'hyper', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        # xform = fk.transforms.AsinhTransform("asinh", param_t=10000, param_m=4.5, param_a=0.25)
        xform = fk.transforms.AsinhTransform(
            "asinh", param_t=10000, param_m=4.5, param_a=0
        )
        s.apply_transform(xform)
        df = pd.DataFrame(s.get_transformed_events(), columns=ch_names)
        # convert time to seconds
        df["Time"] *= float(s.metadata["timestep"])
        df.index.name = "cell"

        # Gate
        xdf = gate_dataframe(
            df,
            gating_strategy=gating_strategies[panel_name],
            plot=plot_gates,
            output_img=output_figure,
        )

        # .iloc[:, 4:-2] <- to remove FSC, Time, etc cols
        xdf.to_csv(output_file)
        try:
            xdf.sample(n=n).to_csv(output_file_subsampled)
        except ValueError:
            xdf.to_csv(output_file_subsampled)

        print(f"Sample '{sample_id}' has {xdf.shape[0]} filtered cells.")


# Concatenate and write H5ad files
meta_m = meta.copy().drop(["other", "flow_comment"], axis=1)

# # since h5ad cannot serialize datetime, let's convert to str
for col in meta_m.columns[meta_m.dtypes == "datetime64[ns]"]:
    meta_m[col] = meta_m[col].astype(str)

for panel_name in gating_strategies:
    panel_dir = output_dir / panel_name

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
        if panel_name == "WB_IgG_IgM":
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
        elif panel_name == "WB_Treg":
            # Channel BV605-A is empty
            df = df.drop("BV605-A", axis=1)

        # x = df.iloc[:, 4:-3]
        x = df.loc[:, ~df.columns.str.contains("FSC|SSC|_ratio|Time|sample")]

        # merge with metadata
        df = df.drop(x.columns, axis=1).merge(
            meta_m, left_on="sample", right_on="sample_name"
        )

        a = AnnData(x, obs=df)
        a.write_h5ad(panel_dir / f"{panel_name}.concatenated.{label}.h5ad")
