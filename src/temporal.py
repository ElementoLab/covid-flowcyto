#!/usr/bin/env python

"""
Longitudinal analysis of COVID19 patients
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from adjustText import adjust_text

import GPy
import scipy
import GPclust

from src.conf import *


def gpy_fit_optimize(x, y, return_model: bool = False):
    if x.shape[0] != 1:
        x = x.reshape((x.size, 1))
        y = y.reshape((y.size, 1))

    output_columns = [
        "D",
        "p_value",
        "mean_posterior_std",
        "RBF",
        "rbf.sum.rbf.variance",
        "rbf.sum.rbf.lengthscale",
        "rbf.sum.bias.variance",
        "rbf.Gaussian_noise.variance",
        "Noise",
        "noise.sum.white.variance",
        "noise.sum.bias.variance",
        "noise.Gaussian_noise.variance",
    ]

    kernel = GPy.kern.RBF(input_dim=1) + GPy.kern.Bias(input_dim=1)
    white_kernel = GPy.kern.White(input_dim=1) + GPy.kern.Bias(input_dim=1)

    m = GPy.models.GPRegression(x, y, kernel)
    try:
        m.optimize()
    except RuntimeWarning:
        return [np.nan] * 11
    w_m = GPy.models.GPRegression(x, y, white_kernel)
    try:
        w_m.optimize()
    except RuntimeWarning:
        return [np.nan] * 11

    # D statistic
    d = 2 * (m.log_likelihood() - w_m.log_likelihood())

    # p-value
    # the RBF + Bias kernel has 4 parameters and the Bias 2, so the chisquare has 2 degrees of freedom is 2
    p = scipy.stats.chi2.sf(d, df=2)

    # Let's calculate the STD of the posterior mean
    # because we have several y inputs for each x value
    # the posterior mean values retrieved will also be duplicated
    # let's make sure our STD is computed on the unique values only
    mean_posterior_std = (
        pd.DataFrame(
            [x.squeeze(), m.posterior.mean.squeeze()], index=["x", "posterior"]
        )
        .T.groupby("x")["posterior"]
        .apply(lambda i: i.unique()[0])
        .std()
    )

    if return_model:
        return m, w_m
    return pd.Series(
        [d, p, mean_posterior_std, m.log_likelihood()]
        + m.param_array.tolist()
        + [w_m.log_likelihood()]
        + w_m.param_array.tolist(),
        index=output_columns,
    )


def fit_MOHGP(x, y, n_clust_guess=4):
    # Make kernels representing underlying process and mean and deviation from it
    k_underlying = GPy.kern.Matern52(
        input_dim=1, variance=1.0, lengthscale=x.max() / 3.0
    )
    k_corruption = GPy.kern.Matern52(
        input_dim=1, variance=0.5, lengthscale=x.max() / 3.0
    ) + GPy.kern.White(1, variance=0.01)

    print("Fitting.")
    model = GPclust.MOHGP(
        X=x,
        Y=y.T,
        kernF=k_underlying,
        kernY=k_corruption,
        K=n_clust_guess,
        alpha=1.0,
        prior_Z="DP",
    )
    model.hyperparam_opt_interval = 1000
    model.hyperparam_opt_args["messages"] = True

    model.optimize(verbose=True)

    print("Finished optimization.")

    # Order clusters by their size
    model.reorder()
    return model


output_dir = results_dir / "temporal"
output_dir.mkdir(exist_ok=True, parents=True)

categories = [
    "severity_group",
    "intubation",
    "death",
]  # , "heme", "bmt", "obesity"]


alpha_thresh = 0.01
log_alpha_thresh = -np.log10(alpha_thresh)


meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file).sort_index(0).sort_index(1)


cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns
parent_population = cols[1].rename("parent_population")


# Get patients with >= 3 timepoints
pts = (
    meta.groupby(["patient", "patient_code"])
    .size()
    .sort_values()
    .loc["Patient"]
)
pts = pts[pts >= 3].index

# pmeta = meta.loc[meta["patient_code"].isin(pts), categories].drop_duplicates()

n = len(matrix.columns)
m = len(categories)
fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 2))

for i, var in enumerate(matrix.columns):
    # var = "LY/All_CD45"
    for patient in pts:
        idx = meta.loc[meta["patient_code"] == patient].index
        x_pmeta = meta.loc[idx].sort_values("datesamples")
        for j, color_cat in enumerate(categories):
            colors = {
                m: sns.color_palette()[i]
                for i, m in enumerate(meta[color_cat].cat.categories)
            }
            cat = x_pmeta.iloc[0][color_cat]
            axes[i, j].plot(
                x_pmeta["datesamples"],
                matrix.loc[x_pmeta.index, var],
                "--o",
                color=colors[cat],
                label=cat,
            )
            axes[i, j].set_ylabel(var)

for i, cat in enumerate(categories):
    axes[0, i].set_title(cat)

for ax in axes[-1, :]:
    ax.set_xlabel("Time since admission (days)")

fig.savefig(
    output_dir / "temporal_timelines.all_variables.svg",
    bbox_inches="tight",
    dpi=300,
    tight_layout=True,
)


# Fit GPs
x = np.log1p(minmax_scale(meta["time_symptoms"].dropna().sort_values()))  # type: ignore
res = pd.concat(
    [
        gpy_fit_optimize(x.values, matrix.loc[x.index, pop].values).rename(pop)
        for pop in matrix.columns
    ],
    axis=1,
).T
res["log_p_value"] = log_pvalues(res["p_value"])
res.to_csv(output_dir / "gaussian_process.fit.csv")

# # plot stats
p = pd.DataFrame(
    dict(x=np.log1p(res["mean_posterior_std"]), y=np.log1p(res["D"]))
)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set(xlabel="log(Mean posterior std)", ylabel="log(D)")
ax.scatter(p["x"], p["y"])
q = p.query("(x > 0.75) | (y > 0.75)")
texts = text(q["x"], q["y"], q.index, ax=ax)
adjust_text(
    texts, arrowprops=dict(arrowstyle="->", color="black"), ax=ax,
)
fig.savefig(output_dir / "gaussian_process.fit.svg", **figkws)

# # plot examples
fig, axis = plt.subplots(1, q.shape[0], figsize=(q.shape[0] * 3, 3))
for i, pop in enumerate(q.index):
    m, w = gpy_fit_optimize(
        x.values, matrix.loc[x.index, pop].values, return_model=True
    )

    m.plot(ax=axis[i], legend=False)
    a = w.plot_f(ax=axis[i], legend=False)
    # a['dataplot'][0].set_color("#e8ab02")
    a["gpmean"][0][0].set_color("#e8ab02")
    a["gpconfidence"][0].set_color("#e8ab02")
    axis[i].set(
        ylabel=pop,
        title=f"D: {res.loc[pop, 'D']:.2f}; p:{res.loc[pop, 'p_value']:.3f}; SD: {res.loc[pop, 'mean_posterior_std']:.2f}",
    )
for ax in axis:
    ax.set_xlabel("Time since symptoms")
fig.savefig(output_dir / "gaussian_process.top_variable.example.svg", **figkws)


# Now do the same for each patient independently
_res = list()

for patient in tqdm(pts):
    x = np.log1p(meta.loc[meta["patient_code"] == patient, "time_symptoms"])
    _res.append(
        pd.concat(
            [
                gpy_fit_optimize(
                    x.values, matrix.loc[x.index, pop].values
                ).rename(pop)
                for pop in matrix.columns
            ],
            axis=1,
        ).T.assign(patient=patient)
    )
res = pd.concat(_res).rename_axis(index="population")
res.to_csv(output_dir / "gaussian_process.per_patient.fit.csv")


res = pd.read_csv(
    output_dir / "gaussian_process.per_patient.fit.csv", index_col=0
)


res["direction"] = res["RBF"] > 0


res_mean = res.groupby(level=0).mean()
res["sig"] = res["p_value"] < 0.05
res_sum = res.groupby(level=0)["sig"].sum()

# # plot stats
p = pd.DataFrame(
    dict(x=np.log1p(res_mean["mean_posterior_std"]), y=np.log1p(res_mean["D"]))
)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set(xlabel="Mean posterior std", ylabel="D")
ax.scatter(p["x"], p["y"])
q = p.query("(x > 1.0) | (y > 0.5)")
texts = text(q["x"], q["y"], q.index, ax=ax)
adjust_text(
    texts, arrowprops=dict(arrowstyle="->", color="black"), ax=ax,
)
fig.savefig(
    output_dir / "gaussian_process.per_patient.aggregated.fit.svg", **figkws
)

#
n_top = 40
n_rows = int(np.ceil(n_top / 5))

fig, axes = plt.subplots(n_rows, 5, figsize=(5 * 3 * 2, n_rows * 3))

for i, (patient, pop) in (
    res.sort_values("D")
    .dropna()
    .tail(n_top)
    .reset_index()[["patient", "population"]]
    .iterrows()
):
    # patient = "P016"
    # pop = "IgM+/LY"
    x = np.log1p(meta.loc[meta["patient_code"] == patient, "time_symptoms"])
    m, w = gpy_fit_optimize(
        x.values, matrix.loc[x.index, pop].values, return_model=True
    )
    m.plot(ax=axes.flat[i], legend=False)
    a = w.plot_f(ax=axes.flat[i], legend=False)
    # a['dataplot'][0].set_color("#e8ab02")
    a["gpmean"][0][0].set_color("#e8ab02")
    a["gpconfidence"][0].set_color("#e8ab02")
    p = res.query(f"patient == '{patient}'")
    axes.flat[i].set(
        ylabel=pop,
        title=f"{patient}; D: {p.loc[pop, 'D']:.2f}; p:{p.loc[pop, 'p_value']:.3f}; SD: {p.loc[pop, 'mean_posterior_std']:.2f}",
    )
for ax in axes.flat:
    ax.set_xlabel("Time since symptoms")
fig.savefig(
    output_dir / "gaussian_process.per_patient.top_variable.example.svg",
    **figkws,
)
plt.close(fig)


# try to summarize metrics across patients somehow
v = (res["p_value"] < 0.1).groupby(res.index).sum().sort_values() / res[
    "patient"
].nunique()
m = res["D"].groupby(res.index).sum().sort_values()


# # Focus on one patient
for pat in res["patient"].unique():
    figure = (
        output_dir / f"gaussian_process.patient_{pat}.top_variable.example.svg"
    )
    if figure.exists():
        continue
    r = res.query(f"patient == '{pat}'")
    n_top = r.shape[0]
    n_rows = int(np.ceil(n_top / 8))
    fig, axes = plt.subplots(n_rows, 8, figsize=(8 * 3 * 2, n_rows * 3))
    for i, (patient, pop) in (
        r.dropna()  # .sort_values("D")
        .tail(n_top)
        .reset_index()[["patient", "population"]]
        .iterrows()
    ):
        x = np.log1p(meta.loc[meta["patient_code"] == patient, "time_symptoms"])
        m, w = gpy_fit_optimize(
            x.values, matrix.loc[x.index, pop].values, return_model=True
        )
        m.plot(ax=axes.flat[i], legend=False)
        a = w.plot_f(ax=axes.flat[i], legend=False)
        # a['dataplot'][0].set_color("#e8ab02")
        a["gpmean"][0][0].set_color("#e8ab02")
        a["gpconfidence"][0].set_color("#e8ab02")
        axes.flat[i].set(
            ylabel=pop,
            title=f"{patient}; D: {r.loc[pop, 'D']:.2f}; p:{r.loc[pop, 'p_value']:.3f}; SD: {r.loc[pop, 'mean_posterior_std']:.2f}",
        )
    for ax in axes.flat:
        ax.set_xlabel("Time since symptoms")
    fig.savefig(
        figure, **figkws,
    )
    plt.close(fig)


#


#


#


#

# Try to cluster cell types based on temporal behaviour using a MOHGP
Y = zscore(matrix.loc[x.index])
model = fit_MOHGP(x.values.reshape((-1, 1)), Y.values)


print("Plotting cluster posteriors.")
# Plot clusters
fig = plt.figure()
model.plot(
    newfig=False,
    on_subplots=True,
    colour=True,
    in_a_row=False,
    joined=False,
    errorbars=False,
)
for ax in fig.axes:
    ax.set_rasterized(True)
    ax.set_ylabel("Population abundance")
    ax.set_xlabel("Time (log2)")
fig.savefig(
    output_dir / "gaussian_process.mohgp.fitted_model.clusters.svg", **figkws
)

print("Plotting parameters/probabilities.")
# Posterior parameters
fig, axis = plt.subplots(
    2,
    1,
    gridspec_kw={"height_ratios": [12, 1]},
    figsize=(3 * 4, 1 * 4),
    tight_layout=True,
)
mat = axis[0].imshow(
    model.phi.T, cmap=plt.get_cmap("hot"), vmin=0, vmax=1, aspect="auto"
)
axis[0].set_xlabel("Region index")
axis[0].set_ylabel("Cluster index")
axis[1].set_aspect(0.1)
plt.colorbar(
    mat, cax=axis[1], label="Posterior probability", orientation="horizontal"
)
fig.savefig(
    output_dir / "gaussian_process.mohgp.fitted_model.posterior_probs.svg",
    **figkws,
)

# Assignment probabilities
g = sns.clustermap(
    model.phi.T,
    cmap=plt.get_cmap("hot"),
    vmin=0,
    vmax=1,
    xticklabels=False,
    rasterized=True,
    figsize=(3, 0.2 * model.phi.T.shape[0]),
    cbar_kws={"label": "Posterior probability"},
)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
g.savefig(
    output_dir
    / "gaussian_process.mohgp.fitted_model.posterior_probs.clustermap.svg",
    **figkws,
)

# Clustermap with cluster assignments
print("Plotting clusters.")
tp = pd.Series(
    matrix.columns.get_level_values("timepoint")
    .str.replace("d", "")
    .astype(int),
    index=matrix.columns,
).sort_values()

g2 = sns.clustermap(
    matrix.loc[x.index],
    col_colors=[plt.get_cmap("Paired")(i) for i in np.argmax(model.phi, 1)],
    row_cluster=False,
    col_cluster=True,
    z_score=1,
    xticklabels=False,
    rasterized=True,
    figsize=(8, 0.2 * matrix.shape[1]),
    metric="correlation",
    robust=True,
)
g2.ax_heatmap.set_xticklabels(
    g2.ax_heatmap.get_xticklabels(), rotation=90, fontsize="xx-small"
)
g2.ax_heatmap.set_yticklabels(g2.ax_heatmap.get_yticklabels(), rotation=0)
g2.ax_col_dendrogram.set_rasterized(True)
g2.savefig(
    output_dir
    / "gaussian_process.mohgp.fitted_model.clustermap.cluster_labels.svg",
    **figkws,
)

matrix_mean = matrix.loc[x.index].groupby(x).mean()
g3 = sns.clustermap(
    matrix_mean,
    col_colors=[plt.get_cmap("Paired")(i) for i in np.argmax(model.phi, 1)],
    row_cluster=False,
    col_cluster=True,
    z_score=1,
    xticklabels=False,
    yticklabels=True,
    rasterized=True,
    figsize=(8, 0.2 * matrix_mean.shape[0]),
    metric="correlation",
    robust=True,
)
g3.ax_heatmap.set_xticklabels(
    g3.ax_heatmap.get_xticklabels(), rotation=90, fontsize="xx-small"
)
g3.ax_heatmap.set_yticklabels(g3.ax_heatmap.get_yticklabels(), rotation=0)
g3.ax_col_dendrogram.set_rasterized(True)
g3.savefig(
    output_dir
    / "gaussian_process.mohgp.fitted_model.mean_acc.clustermap.cluster_labels.svg",
    **figkws,
)
