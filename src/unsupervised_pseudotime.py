#!/usr/bin/env python

"""
Replot some manifolds and get features associated with them.
"""

from sklearn.manifold import SpectralEmbedding
from umap import UMAP
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.conf import *

# from src.unsupervised import plot_projection


def plot_projection(x, meta, cols, n_dims=4, algo_name="PCA"):
    from imc.graphics import to_color_series

    cols = [c for c in cols if c in meta.columns]
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


# Plot arrows between samples of individual patients
df, label1, label2 = (
    zscore(matrix),
    "original",
    "zscore",
)

for name in ["UMAP", "SpectralEmbedding"]:
    prefix = (
        output_dir / f"covid-facs.cell_type_abundances.{name}.{label1}.{label2}"
    )
    model_inst = eval(name)(random_state=0)
    res = pd.DataFrame(model_inst.fit_transform(df), index=df.index)
    if name == "SpectralEmbedding":
        res[0] *= -1

    # Plot projection with only first samples of patients
    meta_r = meta.sort_values("datesamples").drop_duplicates(
        subset=["patient_code"], keep="first"
    )
    fig = plot_projection(res.loc[meta_r.index], meta_r, ["severity_group"], 1)
    fig.savefig(
        prefix + ".only_first_samples.svg", **figkws,
    )
    plt.close(fig)

    meta_r = meta.sort_values("datesamples").drop_duplicates(
        subset=["patient_code"], keep="last"
    )
    fig = plot_projection(res.loc[meta_r.index], meta_r, ["severity_group"], 1)
    fig.savefig(
        prefix + ".only_last_samples.svg", **figkws,
    )
    plt.close(fig)

    # Plot
    pts = (
        meta.groupby(["patient", "patient_code"])
        .size()
        .sort_values()
        .loc["Patient"]
    )
    pts = pts[pts >= 2].index

    res2 = res.set_index(
        res.index.to_series().str.split("-").apply(lambda x: x[0]), append=True
    ).reorder_levels([1, 0])

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(res2[0], res2[1])
    akws = dict(
        width=0.01, head_width=0.15, length_includes_head=True, alpha=0.5,
    )
    colors = sns.color_palette("cubehelix", len(pts))
    for j, pt in enumerate(pts):
        r = res2.loc[pt]
        ax.scatter(r[0], r[1], color=colors[j])
        px = 0
        py = 0
        for i, (x, y) in r.iterrows():
            if i != r.index[0]:
                ax.arrow(px, py, x - px, y - py, color=colors[j], **akws)
            if i == r.index[-1]:
                ax.text(x, y, s=pt)
            px = x
            py = y
    ax.set(xlabel=f"{name}1", ylabel=f"{name}2")
    fig.savefig(
        prefix + ".arrows.svg", **figkws,
    )
    plt.close(fig)

    # See what's related with UMAP1

    if name == "UMAP":
        # standardize
        sres = res.apply(lambda x: (x - x.min()) / (x.max() - x.min())) + 1
        # get x == y
        gradient = np.log(sres[0]) - np.log(sres[1])
    elif name == "SpectralEmbedding":
        gradient = res[0]
    gradient.name = "gradient"
    gradient.to_csv(
        output_dir / f"covid-facs.cell_type_abundances.{name}.gradient.csv"
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(*res.T.values, c=gradient)
    fig.savefig(
        prefix + ".gradient.svg", **figkws,
    )
    plt.close(fig)

    # get axis as
    corr = (
        matrix.join(gradient)
        .corr(method="spearman")["gradient"]
        .drop("gradient")
        .sort_values()
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axhline(0, linestyle="--", color="gray")
    ax.scatter(corr.index, corr)
    # ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=90)
    ax.set(xlabel="Variable", ylabel=f"Spearman correlation\nwith {name} 1")
    fig.savefig(
        prefix + ".correlation_with_gradient.svg", **figkws,
    )
    plt.close(fig)

    # get amount of change in beggining vs end

    # m = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    start = matrix.loc[
        gradient.sort_values().head(gradient.shape[0] // 5).index
    ].mean()
    end = matrix.loc[
        gradient.sort_values().tail(gradient.shape[0] // 5).index
    ].mean()
    change = (end - start).reindex(corr.index)
    log_change = (np.log(end) - np.log(start)).reindex(corr.index)
    mean = matrix.mean().reindex(corr.index)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axhline(0, linestyle="--", color="gray")
    ax.axvline(0, linestyle="--", color="gray")
    ups = corr[corr > 0].index
    dow = corr[corr < 0].index
    # v = log_change.abs().max()
    # v -= v / 20  # cap to 20% of max
    # ax.scatter(log_change, corr, c=mean, cmap="RdBu_r", vmin=-v, vmax=v)
    z1 = ax.scatter(
        log_change.loc[ups],
        corr.loc[ups],
        c=mean.loc[ups],
        cmap="Reds",
        vmin=0,
        vmax=100,
    )
    z2 = ax.scatter(
        log_change.loc[dow],
        corr.loc[dow],
        c=mean.loc[dow],
        cmap="Blues",
        vmin=0,
        vmax=100,
    )
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z1, cax=cax1)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("left", size="5%", pad=0.05)
    fig.colorbar(z2, cax=cax2)
    done = list()
    n = 20
    for t in log_change.sort_values().head(n).index:
        if t not in done:
            ax.text(log_change[t], corr[t], s=t)
            done.append(t)
    for t in corr.sort_values().head(n).index:
        if t not in done:
            ax.text(log_change[t], corr[t], s=t, ha="right")
            done.append(t)
    for t in log_change.sort_values().tail(n).index:
        if t not in done:
            ax.text(log_change[t], corr[t], s=t)
            done.append(t)
    for t in corr.sort_values().tail(n).index:
        if t not in done:
            ax.text(log_change[t], corr[t], s=t, ha="left")
            done.append(t)
    v = log_change.abs().max()
    v += v / 10
    ax.set(
        xlabel="Log(change) pseudotime",
        ylabel="Correlation\n with pseudotime",
        xlim=(-v, v),
    )
    fig.savefig(
        prefix + ".log_change_vs_correlation.svg", **figkws,
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axhline(0, linestyle="--", color="gray")
    ax.axvline(0, linestyle="--", color="gray")
    ups = corr[corr > 0].index
    dow = corr[corr < 0].index
    # v = log_change.abs().max()
    # v -= v / 20  # cap to 20% of max
    # ax.scatter(log_change, corr, c=mean, cmap="RdBu_r", vmin=-v, vmax=v)
    z1 = ax.scatter(
        corr.loc[ups],
        log_change.loc[ups],
        c=mean.loc[ups],
        cmap="Reds",
        vmin=0,
        vmax=100,
    )
    z2 = ax.scatter(
        corr.loc[dow],
        log_change.loc[dow],
        c=mean.loc[dow],
        cmap="Blues",
        vmin=0,
        vmax=100,
    )
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(z1, cax=cax1)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("left", size="5%", pad=0.05)
    fig.colorbar(z2, cax=cax2)
    done = list()
    n = 20
    for t in log_change.sort_values().head(n).index:
        if t not in done:
            ax.text(corr[t], log_change[t], s=t)
            done.append(t)
    for t in corr.sort_values().head(n).index:
        if t not in done:
            ax.text(corr[t], log_change[t], s=t, ha="right")
            done.append(t)
    for t in log_change.sort_values().tail(n).index:
        if t not in done:
            ax.text(corr[t], log_change[t], s=t)
            done.append(t)
    for t in corr.sort_values().tail(n).index:
        if t not in done:
            ax.text(corr[t], log_change[t], s=t, ha="left")
            done.append(t)
    v = corr.abs().max()
    v += v / 10
    ax.set(
        xlabel="Correlation\n with pseudotime",
        ylabel="Log(change) pseudotime",
        xlim=(-v, v),
    )
    fig.savefig(
        prefix + ".correlation_vs_log_change.svg", **figkws,
    )
    plt.close(fig)

    n_top = 16

    fig, axes = plt.subplots(4, 4, figsize=(4 * 4, 2 * 4), tight_layout=True)
    for ax, var in zip(
        axes.flatten(), corr.abs().sort_values().tail(n_top).index
    ):
        ax.scatter(gradient, matrix[var], s=2, alpha=0.5)
        # ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=90)
        child, parent = var.split("/")
        ax.set(
            xlabel=f"{name}1",
            ylabel=f"% {parent}",
            title=f"{child}; r = {corr[var]:.2f}",
        )
    fig.savefig(
        prefix + ".correlated.scatter_with_gradient.svg", **figkws,
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    sns.boxenplot(data=res.join(meta), x=0, y="severity_group", ax=ax)
    for x in ax.get_children():
        if isinstance(x, patches):
            x.set_alpha(0.25)
    sns.swarmplot(data=res.join(meta), x=0, y="severity_group", ax=ax)
    ax.set(xlabel=f"{name}1")
    fig.savefig(
        prefix + ".position_in_gradient_by_severity_group.svg", **figkws,
    )
    plt.close(fig)
