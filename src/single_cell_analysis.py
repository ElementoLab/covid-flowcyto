#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

import re

import scanpy as sc
import pingouin as pg

# from flowsom import flowsom
# from minisom import MiniSom

from imc.graphics import rasterize_scanpy

from src.conf import *


output_dir = Path("results") / "single_cell"
output_dir.mkdir(exist_ok=True, parents=True)

panels = json.load(open(metadata_dir / "flow_variables2.json"))
fcs_dir = data_dir / "fcs"

metadata_file = metadata_dir / "annotation.pq"
meta = pd.read_parquet(metadata_file)


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
label = "full"

for panel_name in gating_strategies:
    panel_dir = output_dir / panel_name
    processed_h5ad = (
        panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"
    )
    processed_h5ad_combat = (
        panel_dir / f"{panel_name}.concatenated.{label}.processed.combat.h5ad"
    )

    prefix = output_dir / f"{panel_name}.{label}."

    if not processed_h5ad_combat.exists():
        a = sc.read_h5ad(panel_dir / f"{panel_name}.concatenated.{label}.h5ad")
        if a.to_df().isnull().any().sum() != 0:
            a = a[~a.to_df().isnull().any(1), :]

        # filter out viability pos
        fig, ax = plt.subplots(1, figsize=(4, 4))
        viabl = "Viability(APC-R700-A)"
        x = a[:, a.var.index == viabl].X
        sns.distplot(x, ax=ax, label=viabl)
        a = a[x < 0][:, ~(a.var.index == viabl)]
        # filter out CD45 neg
        cd45 = "CD45(V500C-A)"
        if cd45 in a.var.index:
            x = a[:, a.var.index == cd45].X
            sns.distplot(x, ax=ax, label=cd45)
            a = a[x > 0][:, ~(a.var.index == cd45)]
        for channel, pop in gating_strategies[panel_name][1:]:
            x = a[:, a.var.index.str.contains(channel, regex=False)].X
            sns.distplot(x, ax=ax, label=channel)
        ax.legend()
        fig.savefig(prefix + "distributions.svg", **figkws)
        plt.close(fig)

        # Process
        sc.pp.pca(a)
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a)
        sc.tl.leiden(a, resolution=0.025, key_added="cluster")
        a.write_h5ad(processed_h5ad)

        sc.pp.combat(a, "sample_id")
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a)
        sc.tl.leiden(a, resolution=0.1, key_added="cluster")
        a.write_h5ad(processed_h5ad_combat)

    # Read in
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

    if panel in ["WB_Memory", "WB_Checkpoint", "WB_Treg"]:
        fig, ax = plt.subplots(1, figsize=(4, 4))
        ax.scatter(
            a[:, a.var.index.str.startswith(r"CD4(")].X,
            a[:, a.var.index.str.startswith("CD8")].X,
            s=1,
            alpha=0.1,
            rasterized=True,
        )
        ax.set(xlabel="CD4", ylabel="CD8")
        fig.savefig(
            prefix + "CD4_vs_CD8.svg", dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    prefix += "combat."
    a = sc.read(processed_h5ad_combat)
    # randomize order prior to plotting
    a = a[a.to_df().iloc[:, 0].sample(frac=1).index, :]

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
        / f"{panel_name}.{label}.cluster_mean_intensity.clustermap.zscore.svg",
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

    # Stacked barplots
    for var in ["severity_group", "sex", "death", "tocilizumab"]:
        df = (
            a.obs.groupby("cluster")[var]
            .value_counts(normalize=True)
            .rename("percentage")
            .reset_index()
            .pivot_table(
                index="cluster", columns=var, values="percentage", fill_value=0,
            )
        )
        # gotta normalize by the expected given the number of cells in each group
        df = df * (1 / (a.obs.groupby(var).size() / a.shape[0]))
        df = (df.T / df.sum(1)).T * 100

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        n = meta[var].nunique()
        center = 100 / n
        sns.heatmap(
            df,
            cbar_kws=dict(label="% of cells"),
            ax=ax,
            center=center,
            cmap="RdBu_r",
            vmax=100,
            vmin=0,
        )
        fig.savefig(
            prefix + f"cluster_composition.clustermap.colored_by_{var}.svg",
            **figkws,
        )

        fig, ax = plt.subplots(n, 1, sharey=True, sharex=False, figsize=(4, 8))
        ctrl = meta[var].min()
        for i, val in enumerate(meta[var].dropna().unique()):
            lfc = np.log2((df[val] / df[ctrl]).T).sort_values()
            ax[i].scatter(lfc.index, lfc)
            ax[i].axhline(0, linestyle="--", color="grey")
            ax[i].set(title=val, xlabel="Cluster", ylabel="log2(fold-change")
        fig.savefig(
            prefix + f"cluster_composition.rank_vs_fold_change.by_{var}.svg",
            **figkws,
        )

        # Investigate cluster phenotypes
        mean = a.to_df().groupby(a.obs["cluster"]).mean()
        count = a.obs["cluster"].value_counts().rename("Cells per cluster")
        kws = dict(
            row_colors=df.join(count),
            col_colors=a.to_df().mean().rename("Channel mean"),
            figsize=(6, 6),
        )
        grid = sns.clustermap(
            mean, cbar_kws=dict(label="Mean intensity"), **kws
        )
        grid.savefig(
            prefix + f"cluster_mean_intensity.clustermap.colored_by_{var}.svg",
            **figkws,
        )

    fold_change = (df.T / df["negative"]).T.drop("negative", 1)
    # log_fold_change = np.log2(fold_change)

    sns.heatmap(df)
    sns.heatmap(df, center=20, cmap="RdBu_r", robust=True)
    sns.heatmap(fold_change, center=1, cmap="RdBu_r", robust=True)

    # Get patients with >= 3 timepoints
    pts = (
        meta.groupby(["patient", "patient_code"])
        .size()
        .sort_values()
        .loc["Patient"]
    )
    pts = pts[pts >= 3].index

    pat = "P016"

    p = a[a.obs.query(f"patient_code == '{pat}'").index, :]

    times = sorted(p.obs["time_symptoms"].unique())
    n_cols = 1 + len(times)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    k = dict(show=False, size=4)
    sc.pl.umap(p, color=["time_symptoms"], cmap="rainbow", ax=axes[0], **k)
    for i, time in enumerate(times, 1):
        p.obs["plot"] = (p.obs["time_symptoms"] == time).astype(float)
        print(p.obs["plot"].sum())
        sc.pl.umap(p, color=["plot"], cmap="Reds", ax=axes[i], vmin=-0.25, **k)
        axes[i].set_title(time)
    rasterize_scanpy(fig)
    fig.savefig(prefix + f"patient_{pat}.global_projection.svg", **figkws)

    # Compare with newly designed space
    sc.pp.combat(p, "processing_batch_categorical")
    sc.pp.pca(p)
    sc.pp.neighbors(p, n_neighbors=50)
    sc.tl.umap(p)
    sc.tl.leiden(p, resolution=0.025, key_added="cluster")

    n_cols = max(n_cols, p.shape[1])

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 2 * 4))
    for i, ch in enumerate(p.var.index):
        sc.pl.umap(p, color=[ch], ax=axes[0, i], show=False)
    k = dict(show=False, size=4)
    sc.pl.umap(p, color=["time_symptoms"], cmap="rainbow", ax=axes[1, 0], **k)
    for i, time in enumerate(times, 1):
        p.obs["plot"] = (p.obs["time_symptoms"] == time).astype(float)
        print(p.obs["plot"].sum())
        sc.pl.umap(
            p, color=["plot"], cmap="Reds", ax=axes[1, i], vmin=-0.25, **k
        )
        axes[1, i].set_title(time)
    rasterize_scanpy(fig)
    fig.savefig(prefix + f"patient_{pat}.own_projection.combat.svg", **figkws)

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
