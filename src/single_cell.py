#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

import re

import flowkit as fk
from anndata import AnnData
import scanpy as sc
import pingouin as pg

# from flowsom import flowsom
# from minisom import MiniSom

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


# Actually run analysis
clin_vars = [
    "patient_code",
    "COVID19",
    "severity_group",
    "sex",
    "age",
    "bmi",
    "tocilizumab",
]
# panel = "WB_IgG_IgM"
# panel = "WB_Memory"
# label = "subsampled"
# label = "full"

for panel in pos_gate_names:
    panel_dir = output_dir / panel
    processed_h5ad = panel_dir / f"{panel}.concatenated.{label}.processed.h5ad"

    prefix = output_dir / f"{panel}.{label}."

    if not processed_h5ad.exists():
        a = sc.read_h5ad(panel_dir / f"{panel}.concatenated.{label}.h5ad")
        if a.to_df().isnull().any().sum() != 0:
            a = a[~a.to_df().isnull().any(1), :].copy()

        # filter out viability pos
        fig, ax = plt.subplots(1, figsize=(4, 4))
        viabl = "Viability(APC-R700-A)"
        x = a[:, a.var.index == viabl].X
        sns.distplot(x, ax=ax, label=viabl)
        a = a[x < 0][:, ~(a.var.index == viabl)].copy()
        # filter out CD45 neg
        cd45 = "CD45(V500C-A)"
        if cd45 in a.var.index:
            x = a[:, a.var.index == cd45].X
            sns.distplot(x, ax=ax, label=cd45)
            a = a[x > 0][:, ~(a.var.index == cd45)].copy()
        pos_name = pos_gate_names[panel]
        x = a[:, a.var.index.str.contains(pos_name)].X
        sns.distplot(x, ax=ax, label=pos_name)
        ax.legend()
        fig.savefig(prefix + "distributions.svg", **figkws)
        plt.close(fig)

        # Process
        sc.pp.pca(a)
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a)
        sc.tl.leiden(a, resolution=0.025, key_added="cluster")
        a.write_h5ad(panel_dir / f"{panel}.concatenated.{label}.processed.h5ad")

        sc.pp.combat(a, "sample_id")
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a)
        sc.tl.leiden(a, resolution=0.1, key_added="cluster")
        a.write_h5ad(
            panel_dir / f"{panel}.concatenated.{label}.processed.combat.h5ad"
        )

        prefix += "combat."

    a = sc.read(processed_h5ad)
    if panel == "WB_IgG_IgM":
        fig, ax = plt.subplots(1, figsize=(4, 4))
        ax.scatter(
            a[:, "CD20(APC-H7-A)"].X,
            a[:, "CD19(Pacific Blue-A)"].X,
            s=1,
            alpha=0.1,
            rasterized=True,
        )
        ax.set(xlabel="CD20(APC-H7-A)", ylabel="CD19(Pacific Blue-A)")
        fig.savefig(
            prefix + "CD20_vs_CD19.svg", dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    if panel in ["WB_Memory", "WB_Checkpoint"]:
        fig, ax = plt.subplots(1, figsize=(4, 4))
        ax.scatter(
            a[:, "CD4(BV605-A)"].X,
            a[:, "CD8(APC-H7-A)"].X,
            s=1,
            alpha=0.1,
            rasterized=True,
        )
        ax.set(xlabel="CD4(BV605-A)", ylabel="CD8(APC-H7-A)")
        fig.savefig(
            prefix + "CD4_vs_CD8.svg", dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    # to subsample:
    # a = a[a.to_df().sample(n=25_000).index, :].copy()

    # randomize order prior to plotting

    a = a[a.to_df().iloc[:, 0].sample(frac=1).index, :]

    from imc.graphics import rasterize_scanpy

    f = prefix + "pca.svg"
    if not f.exists():
        fig = sc.pl.pca(
            a,
            color=a.var.index.tolist() + clin_vars + ["cluster"],
            show=False,
            return_fig=True,
        )
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)
        plt.close(fig)

    f = prefix + "umap.svg"
    if not f.exists():
        fig = sc.pl.umap(
            a,
            color=a.var.index.tolist() + clin_vars + ["cluster"],
            show=False,
            return_fig=True,
        )
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)
        plt.close(fig)

    # Investigate cluster phenotypes
    mean = a.to_df().groupby(a.obs["cluster"]).mean()
    count = a.obs["cluster"].value_counts().rename("Cells per cluster")
    kws = dict(
        row_colors=count,
        col_colors=a.to_df().mean().rename("Channel mean"),
        figsize=(6, 6),
    )
    grid = sns.clustermap(mean, cbar_kws=dict(label="Mean intensity"), **kws)
    grid.savefig(
        prefix + "cluster_mean_intensity.clustermap.svg", **figkws,
    )
    grid = sns.clustermap(
        mean,
        z_score=1,
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="Mean intensity\n(Z-score)"),
        **kws,
    )
    grid.savefig(
        output_dir
        / f"{panel}.{label}.cluster_mean_intensity.clustermap.zscore.svg",
        **figkws,
    )

    # # name clusters as cell types

    # Compare abundance of cells between factors
    totest = (
        a.obs.groupby("sample_id")["cluster"]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["cluster", "sample_id"])
        .reset_index(drop=True)
    )
    # fill in zeros (this is important for testing!)
    totest = (
        totest.pivot_table(
            index="sample_id", columns="cluster", fill_value=0, values="count"
        )
        .reset_index()
        .melt(id_vars=["sample_id"], var_name="cluster", value_name="count")
    )
    totest = totest.join(
        totest.groupby("sample_id")
        .apply(lambda x: (x["count"] / x["count"].sum()) * 100)
        .rename("abundance (%)")
        .reset_index(drop=True)
    )
    totest = totest.merge(
        a.obs[categories + ["sample_id"]].drop_duplicates(), on="sample_id"
    )

    _test_res = list()
    for cat_var in categories:
        control = meta[cat_var].min()

        # Test for differences
        _aov = list()
        for cluster in totest["cluster"].unique():
            for var in totest[cat_var].dropna().unique():
                if var == control:
                    continue
                _a = totest.query(
                    f"cluster == '{cluster}' and {cat_var} == '{var}'"
                )["abundance (%)"]
                _b = totest.query(
                    f"cluster == '{cluster}' and {cat_var} == '{control}'"
                )["abundance (%)"]
                try:
                    t = pg.mwu(_a, _b).assign(
                        cluster=cluster, variable=cat_var, value=var
                    )
                except (AssertionError, ValueError):
                    continue
                _aov.append(t)
        if _aov:
            aov = pd.concat(_aov).set_index("cluster")
            aov["fdr"] = pg.multicomp(aov["p-val"].values, method="fdr_bh")[1]
            _test_res.append(aov)

        kws = dict(
            data=totest[["cluster", cat_var, "abundance (%)"]],
            x=cat_var,
            y="abundance (%)",
            hue=cat_var,
            palette="tab10",
        )
        grid = sns.FacetGrid(
            data=totest[["cluster", cat_var, "abundance (%)"]],
            col="cluster",
            sharey=False,
            height=3,
            col_wrap=4,
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
        group = (
            re.findall(r"^.*\[T.(.*)\]", cat_var)[0]
            if "[" in cat_var
            else cat_var
        )
        for ax in grid.axes.flat:
            var = ax.get_title().replace("cluster = ", "")
            pop = var
            try:
                pop, parent = re.findall(r"(.*)/(.*)", pop)[0]
                ax.set_ylabel(f"% {parent}")
            except IndexError:
                pass
            ax.set_title(
                pop
                + f"\n{group}/{control}:\n"
                + f"FDR = {aov.loc[pop, 'fdr'].min():.3e}"  # + f"Coef = {aov.loc[pop, 'F']:.3f}; "
            )
        grid.savefig(
            prefix + f"{cat_var}.cluster_abundance_comparisons.svg", **figkws,
        )
        plt.close(grid.fig)

    test_res = pd.concat(_test_res).rename(columns={"Source": "factor"})
    test_res.to_csv(prefix + "mwu.cluster_abundance_comparisons.csv")

    # # Compare differences in expression between factors
    # res = a.to_df().join(a.obs)
    # res.to_parquet("")

    # # Try to use a subsample of cells to learn the UMAP manifold
    # import umap
    # b = a[a.to_df().sample(n=10_000).index, :].copy()
    # model = umap.UMAP(n_neighbors=3, random_state=42, min_dist=0.5).fit(b.X.astype(float))

    # train_embedding = model.transform(b.X)
    # test_embedding = model.transform(a.X)

    # fig, axes = plt.subplots(1, 2)
    # axes[0].scatter(test_embedding[:, 0], test_embedding[:, 1], color="orange", alpha=0.5, s=2)
    # axes[0].scatter(train_embedding[:, 0], train_embedding[:, 1], color="blue", alpha=0.5, s=2)
