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
    # "COVID19",
    "severity_group",
    "hospitalization",
    "intubation",
    "death",
    "sex",
    # "race",
    "age",
    "bmi",
    "tocilizumab",
]
# panel_name = "WB_IgG_IgM"
# panel_name = "WB_Memory"
# panel_name = 'PBMC_MDSC'
# panel_name = "WB_T3"
# panel_name = "WB_Treg"
# label = "subsampled"
label = "full"


for panel_name in gating_strategies:
    panel_dir = output_dir / panel_name
    processed_h5ad = (
        panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"
    )
    # processed_h5ad_combat = (
    #     panel_dir / f"{panel_name}.concatenated.{label}.processed.combat.h5ad"
    # )

    prefix = output_dir / f"{panel_name}.{label}."

    if not processed_h5ad.exists():
        a = sc.read_h5ad(panel_dir / f"{panel_name}.concatenated.{label}.h5ad")
        # a = a[a.obs["severity_group"] != "non-covid", :]
        if a.to_df().isnull().any().sum() != 0:
            a = a[~a.to_df().isnull().any(1), :]

        # filter out viability pos
        fig, ax = plt.subplots(1, figsize=(4, 4))
        viabl = "Viability(APC-R700-A)"
        x = a[:, a.var.index == viabl].X
        sns.distplot(x, ax=ax, label=viabl)
        a = a[x < 0]
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

        # plot post-filtering
        fig, ax = plt.subplots(1, figsize=(4, 4))
        viabl = "Viability(APC-R700-A)"
        x = a[:, a.var.index == viabl].X
        sns.distplot(x, ax=ax, label=viabl)
        cd45 = "CD45(V500C-A)"
        if cd45 in a.var.index:
            x = a[:, a.var.index == cd45].X
            sns.distplot(x, ax=ax, label=cd45)
        for channel, pop in gating_strategies[panel_name][1:]:
            x = a[:, a.var.index.str.contains(channel, regex=False)].X
            sns.distplot(x, ax=ax, label=channel)
        ax.legend()
        fig.savefig(prefix + "distributions.post_filtering.svg", **figkws)
        plt.close(fig)

        # Remove viability channel
        a = a[:, ~(a.var.index == viabl)]

        # Process
        sc.pp.pca(a)
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a)
        fac = 1.0
        if panel_name == "WB_Memory":
            fac = 2.0
        elif panel_name == "WB_IgG_IgM":
            fac = 5.0
        elif panel_name == "WB_NK_KIR":
            fac = 2.5
        elif panel_name == "PBMC_MDSC":
            fac = 2.0
        elif panel_name == "WB_T3":
            fac = 2.0
        sc.tl.leiden(a, resolution=0.025 * fac, key_added="cluster")
        a.write_h5ad(processed_h5ad)

        # sc.pp.combat(a, "sample_id")
        # sc.pp.neighbors(a, n_neighbors=15)
        # sc.tl.umap(a)
        # sc.tl.leiden(a, resolution=0.1, key_added="cluster")
        # a.write_h5ad(processed_h5ad_combat)

    # Read in
    a = sc.read(processed_h5ad)
    a = a[a.to_df().iloc[:, 0].sample(frac=1).index, :]

    if panel_name == "WB_IgG_IgM":
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

    if panel_name in ["WB_Memory", "WB_Checkpoint", "WB_Treg", "WB_T3"]:
        fig, ax = plt.subplots(1, figsize=(4, 4))
        ax.scatter(
            a[:, a.var.index.str.startswith(r"CD4(")].X,
            a[:, a.var.index.str.startswith("CD8")].X,
            s=0.5,
            alpha=0.1,
            rasterized=True,
        )
        ax.set(xlabel="CD4", ylabel="CD8")
        fig.savefig(
            prefix + "CD4_vs_CD8.svg", dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    # prefix += "combat."
    # a = sc.read(processed_h5ad_combat)
    # randomize order prior to plotting
    # a = a[a.to_df().iloc[:, 0].sample(frac=1).index, :]

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
        for clin_var in clin_vars:
            fig = sc.pl.umap(
                a,
                color=[clin_var],
                palette=palettes.get(clin_var),
                show=False,
                return_fig=True,
            )
            rasterize_scanpy(fig)
            fig.savefig(str(f).replace(".svg", f".{clin_var}.svg"), **figkws)
            plt.close(fig)

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
        center=0,
        figsize=(6, 6),
        xticklabels=True,
        yticklabels=True,
    )
    grid1 = sns.clustermap(
        mean, cbar_kws=dict(label="Mean intensity"), cmap="Spectral_r", **kws
    )
    grid1.savefig(
        prefix + "cluster_mean_intensity.clustermap.svg", **figkws,
    )
    plt.close(grid1.fig)
    grid2 = sns.clustermap(
        mean,
        z_score=1,
        cmap="RdBu_r",
        cbar_kws=dict(label="Mean intensity\n(Z-score)"),
        **kws,
    )
    grid2.savefig(
        output_dir
        / f"{panel_name}.{label}.cluster_mean_intensity.clustermap.zscore.svg",
        **figkws,
    )
    plt.close(grid2.fig)

    # Association between factors and cluster distribution
    for var in clin_vars[1:]:
        y = a.obs[[var]].join(pd.get_dummies(a.obs["cluster"]))
        v = dict()
        for cat in a.obs["cluster"].unique():
            v[cat] = pg.chi2_independence(data=y, x=var, y=cat)[2].iloc[-1]
        res = pd.DataFrame(v).T

        # same order as clustermap
        res = res.reindex(mean.iloc[grid1.dendrogram_row.reordered_ind].index)

        fig, ax = plt.subplots(1, 1, figsize=(1.430, 0.08))
        cramer = res["cramer"].astype(float)
        points = ax.scatter(
            cramer.index,
            [0] * cramer.shape[0],
            s=6,
            c=cramer,
            cmap="autumn_r",
            marker="s",
            edgecolors="none",
            vmax=0.8,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")
        fig.colorbar(points, label="Cramer's V")
        fig.savefig(
            output_dir
            / f"{panel_name}.{label}.cluster_composition_association_with_{var}.cramer.svg",
            **figkws,
        )

        df = (
            a.obs.groupby("cluster")[var]
            .value_counts(normalize=True)
            .rename("percentage")
            .reset_index()
            .pivot_table(
                index="cluster", columns=var, values="percentage", fill_value=0,
            )
        )
        # reorder as cluster above
        # reorder as original category
        df = df.loc[
            mean.iloc[grid1.dendrogram_row.reordered_ind].index,
            meta[var].cat.categories,
        ]
        pal = palettes[var]

        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        for i, col in enumerate(df.columns):
            c = pal[meta[var].cat.categories.tolist().index(col)]
            if i == 0:
                cum = None
            ax.bar(df.index, df[col], color=c, bottom=cum)
            if i == 0:
                cum = df[col]
            else:
                cum = df[col] + cum
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig.savefig(
            prefix + f"cluster_composition.stacked_bar.by_{var}.svg", **figkws,
        )
        plt.close(fig)

        # normalize by the expected given the number of cells in each group?
        comp = a.obs.groupby(var).size() / a.shape[0]
        df2 = df / comp
        df2 = (df2.T / df2.sum(1)).T

        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        for i, col in enumerate(df2.columns):
            c = pal[meta[var].cat.categories.tolist().index(col)]
            if i == 0:
                cum = None
            ax.bar(df2.index, df2[col], color=c, bottom=cum)
            if i == 0:
                cum = df2[col].copy()
            else:
                cum = cum + df2[col]
        fig.savefig(
            prefix + f"cluster_composition.stacked_bar.by_{var}.normalized.svg",
            **figkws,
        )
        plt.close(fig)

    #

    #

    #

    #

    #

    # Below is experimental:

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

    fig, ax = plt.subplots(1, 1, figsize=(2, 4))
    counts = p.obs["time_symptoms"].value_counts()
    sns.barplot(counts.index, counts.values)
    fig.savefig(prefix + f"patient_{pat}.cells_per_time.svg", **figkws)

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

    var = "time_symptoms"
    df = (
        p.obs.groupby("cluster")[var]
        .value_counts(normalize=True)
        .rename("percentage")
        .reset_index()
        .pivot_table(
            index="cluster", columns=var, values="percentage", fill_value=0,
        )
    )
    cmap = plt.get_cmap("Reds")
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    for i, col in enumerate(df.columns):
        if i == 0:
            cum = None
        ax.bar(
            df.index, df[col], bottom=cum, color=cmap(col / df.columns.max())
        )
        if i == 0:
            cum = df[col]
        else:
            cum += df[col]
    fig.savefig(
        prefix + f"patient_{pat}.global_projection.stacked_bar_by_{var}.svg",
        **figkws,
    )

    # Compare with newly designed space
    sc.pp.combat(p, "processing_batch_categorical")
    sc.pp.pca(p)
    sc.pp.neighbors(p, n_neighbors=50)
    sc.tl.umap(p)
    sc.tl.leiden(p, resolution=0.5, key_added="cluster")

    sc.write(prefix + f"{pat}.own_projection.processed.h5ad", p)

    n_cols = max(n_cols, p.shape[1]) + 1

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 2 * 4))
    for i, ch in enumerate(p.var.index):
        sc.pl.umap(p, color=[ch], ax=axes[0, i], show=False)
    k = dict(show=False, size=4)
    sc.pl.umap(p, color=["cluster"], cmap="rainbow", ax=axes[1, 0], **k)
    sc.pl.umap(p, color=["time_symptoms"], cmap="rainbow", ax=axes[1, 1], **k)
    for i, time in enumerate(times, 2):
        p.obs["plot"] = (p.obs["time_symptoms"] == time).astype(float)
        print(p.obs["plot"].sum())
        sc.pl.umap(
            p, color=["plot"], cmap="Reds", ax=axes[1, i], vmin=-0.25, **k
        )
        axes[1, i].set_title(time)
    rasterize_scanpy(fig)
    fig.savefig(prefix + f"patient_{pat}.own_projection.combat.svg", **figkws)

    var = "time_symptoms"
    df = (
        p.obs.groupby("cluster")[var]
        .value_counts(normalize=True)
        .rename("percentage")
        .reset_index()
        .pivot_table(
            index="cluster", columns=var, values="percentage", fill_value=0,
        )
    )
    cmap = plt.get_cmap("Reds")
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    for i, col in enumerate(df.columns):
        if i == 0:
            cum = None
        ax.bar(
            df.index, df[col], bottom=cum, color=cmap(col / df.columns.max())
        )
        if i == 0:
            cum = df[col]
        else:
            cum += df[col]
    fig.savefig(
        prefix + f"patient_{pat}.own_projection.stacked_bar_by_{var}.svg",
        **figkws,
    )
    # # Compare differences in expression between factors

    # Get patients with >= 3 timepoints
    import umap

    pts = (
        meta.groupby(["patient", "patient_code"])
        .size()
        .sort_values()
        .loc["Patient"]
    )
    pts = pts[pts >= 3].index

    pat = "P016"
    p = a[a.obs.query(f"patient_code == '{pat}'").index, :]
    p.obs["time_class"] = p.obs["time_symptoms"].astype(int).astype(str)

    # Get balanced number of cells per timepoint
    n = p.obs["time_symptoms"].value_counts().min()
    times = p.obs["time_symptoms"].unique()

    idx = [
        i
        for t in times
        for i in p.obs.query(f"time_symptoms == {t}").sample(n=n).index.tolist()
    ]
    assert len(idx) == n * len(times)

    # fit manifold with balanced data
    trans = umap.UMAP(n_neighbors=15, random_state=42).fit(p[idx].X)

    original = p.obsm["X_umap"].copy()
    p.obsm["X_umap"] = trans.transform(p.X)

    sc.pl.umap(p, color=p.var.index.tolist() + ["time_class"])

    # Quantify expression over time
    pts = (
        meta.groupby(["patient", "patient_code"])
        .size()
        .sort_values()
        .loc["Patient"]
    )
    pts = pts[pts >= 3].index

    for pat in pts:
        p = a[a.obs.query(f"patient_code == '{pat}'").index, :]

        pv = p.to_df().drop("CD3(FITC-A)", axis=1)
        mean = (
            pv.join(p.obs["time_symptoms"])
            .groupby("time_symptoms")
            .mean()
            .mean()
        )
        mean = pv.mean()
        pv = (pv - mean) / pv.std()
        pv = pv.join(p.obs["time_symptoms"]).sort_values("time_symptoms")
        pm = pv.groupby("time_symptoms").mean()

        fig, ax = plt.subplots(1, 1, figsize=(1.5, 3))
        sns.heatmap(pm.T, cmap="RdBu_r", cbar_kws=dict(label="Z-score"), ax=ax)
        fig.suptitle(pat)
        fig.savefig(
            prefix + f"patient_{pat}.change_over_time.heatmap.svg", **figkws,
        )
        plt.close(fig)

        pv["time_class"] = pv["time_symptoms"].astype(int).astype(str)
        grid = sns.catplot(
            kind="bar",
            data=pv.drop("time_symptoms", axis=1).melt(id_vars="time_class"),
            col="variable",
            y="value",
            x="time_class",
            order=pv["time_class"].unique(),
            palette="Reds",
            sharey=False,
            height=2,
            aspect=0.8,
        )
        for ax in grid.axes.flat:
            y = abs(np.asarray(ax.get_ylim())).max()
            if y < 0.5:
                ax.set_ylim((-0.5, 0.5))
            else:
                ax.set_ylim((-y, y))
        grid.fig.suptitle(pat)
        grid.fig.savefig(
            prefix + f"patient_{pat}.change_over_time.barplot.svg", **figkws,
        )
        plt.close(grid.fig)
