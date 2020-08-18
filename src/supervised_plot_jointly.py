#!/usr/bin/env python

"""
This script plots the estimates jointly for all models.
"""

import networkx as nx  # type: ignore

from src.conf import *


output_dir = results_dir / "supervised"
output_dir.mkdir(exist_ok=True, parents=True)


# Display all effects together
reduction = "reduced"
ks = dict(center=0, cmap="RdBu_r", robust=True, metric="correlation")

# # First for all variables (no filtering)
for i, (model_name, model) in enumerate(list(models.items())):
    # break
    prefix = f"differential.{model_name}.{reduction}."
    res = pd.read_csv(
        output_dir / f"differential.{model_name}.{reduction}.results.csv"
    )
    long_f = res.pivot_table(index="variable", columns="comparison")
    _coefs = long_f["coef"].drop("Intercept", 1)
    # _coefs = _coefs.loc[:, ~_coefs.columns.str.contains("batch")]
    for variable in model["continuous"]:
        _coefs[variable] = _coefs[variable] * (meta[variable].mean() / 2)
    _coefs.columns = _coefs.columns + "___" + model_name
    _pvals = long_f["qval"].drop("Intercept", 1)
    _pvals.columns = _pvals.columns + "___" + model_name
    if i == 0:
        coefs = _coefs
        pvals = _pvals
    else:
        coefs = coefs.join(_coefs)
        pvals = pvals.join(_pvals)

    # stricter version
    res = res.loc[res["llf"] < np.inf]
    long_f = res.pivot_table(index="variable", columns="comparison")
    # drop columns with levels not estimated
    long_f = long_f.loc[
        :,
        long_f.columns.get_level_values(1).isin(
            long_f["coef"].columns[long_f["coef"].abs().sum() > 1e-10]
        ),
    ]
    _coefs = long_f["coef"].drop("Intercept", 1)
    for variable in model["continuous"]:
        _coefs[variable] = _coefs[variable] * (meta[variable].mean() / 2)
    _coefs.columns = _coefs.columns + "___" + model_name
    _pvals = long_f["qval"].drop("Intercept", 1)
    _pvals.columns = _pvals.columns + "___" + model_name
    if i == 0:
        coefs_strict = _coefs
        pvals_strict = _pvals
    else:
        coefs_strict = coefs_strict.join(_coefs)
        pvals_strict = pvals_strict.join(_pvals)


# # Expand
# for col in coefs.columns[
#     coefs.columns.str.startswith("severity_group[T.severe]")
# ]:
#     factor, model = col.split("___")
#     x = "severity_group[T.mild]___" + model
#     name = "severity_group[T.severe/T.mild]___" + model
#     try:
#         coefs[name] = coefs[col] - coefs[x]
#         break
#     except KeyError:
#         pass


# # Make version with coefficients reduced by mean if appearing in more than one model


coefs = coefs.loc[:, ~coefs.columns.str.contains("COVID19")]
pvals = pvals.loc[:, ~pvals.columns.str.contains("COVID19")]
coefs_strict = coefs_strict.loc[
    :, ~coefs_strict.columns.str.contains("COVID19")
]
pvals_strict = pvals_strict.loc[
    :, ~pvals_strict.columns.str.contains("COVID19")
]


c = coefs.columns.to_series().str.split("___").apply(pd.Series)
c.columns = ["var", "model"]
coefs_red = coefs.T.groupby(c["var"]).mean().T

c = coefs_strict.columns.to_series().str.split("___").apply(pd.Series)
c.columns = ["var", "model"]
coefs_strict_red = coefs_strict.T.groupby(c["var"]).mean().T

c = pvals.columns.to_series().str.split("___").apply(pd.Series)
c.columns = ["var", "model"]
pvals_red = (pvals < 0.05).T.groupby(c["var"]).any().T

c = pvals_strict.columns.to_series().str.split("___").apply(pd.Series)
c.columns = ["var", "model"]
pvals_strict_red = (pvals_strict < 0.05).T.groupby(c["var"]).any().T


# # Plot both versions
prefix = output_dir / "differential.all_models.coefficients"
k = dict(
    cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
    xticklabels=True,
    yticklabels=True,
)
for co, p, label in [
    (coefs, pvals < 0.05, "all_vars"),
    (coefs_strict, pvals_strict < 0.05, "strict_vars"),
    (coefs_red, pvals_red, "reduced.all_vars"),
    (coefs_strict_red, pvals_strict_red, "reduced.strict_vars"),
]:
    print(co.isnull().sum().sum())
    p.index.name = co.index.name

    co = co.loc[:, ~co.columns.str.contains("batch")]
    p = p.loc[:, ~p.columns.str.contains("batch")]

    grid1 = sns.clustermap(co.fillna(0).T, figsize=(18, 10), **ks, **k)
    grid1.savefig(prefix + f".{label}.clustermap.svg")
    plt.close(grid1.fig)

    # Plot version where insignificant changes are masked
    grid2 = sns.clustermap(
        co.fillna(0).T, figsize=(18, 10), mask=~p.T, **ks, **k
    )
    grid2.savefig(prefix + f".{label}.clustermap.sig_masked.svg")
    plt.close(grid2.fig)

    # plot only significance as mask, clustered as above
    grid3 = sns.clustermap(
        p.T,
        row_linkage=grid1.dendrogram_row.linkage,
        col_linkage=grid1.dendrogram_col.linkage,
        figsize=(18, 10),
        **k,
        cmap="gray",
        cbar_pos=None,
        mask=~p.T,
    )
    grid3.savefig(prefix + f".{label}.clustermap.sig_only.svg")
    plt.close(grid3.fig)

    # plot only significance as mask, clustered as above
    g = sns.clustermap(
        co.corr(),
        figsize=(10, 10),
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
    )
    g.savefig(prefix + f".{label}.correlation.clustermap.svg")
    plt.close(g.fig)

    # plot only significance as mask, clustered as above
    g = sns.clustermap(
        co.T.corr(),
        figsize=(10, 10),
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
    )
    g.savefig(prefix + f".{label}.variable_correlation.clustermap.svg")
    plt.close(g.fig)

    # Make network for visualization
    c = co.corr()
    c = c[(c < -0.3) | (c > 0.3)]
    c.iloc[0, 0] = -1  # this is just to scale -1 to 1
    # rescale -1:1 to 0:1
    c = (c - np.nanmin(c)) / (np.nanmax(c) - np.nanmin(c))
    interactions = (
        c.rename_axis(index="v").reset_index().melt(id_vars=["v"]).dropna()
    )
    interactions.columns = ["source", "target", "value"]
    interactions["edgetype"] = "factor-factor"

    _edges = list()
    for factor in p.columns:
        x = p[factor]
        for var in x[x].index:
            _edges.append([factor, var, co.loc[var, factor]])
    edges = pd.DataFrame(_edges, columns=["source", "target", "value"]).assign(
        edgetype="factor-pop"
    )
    # scale to 0:1
    v = edges["value"]
    t = v.abs().max()
    edges["value"] = (v - -t) / (t - -t)

    edgelist = interactions.append(edges)
    edgelist = edgelist.loc[edgelist["source"] != edgelist["target"]]
    edgelist.loc[edgelist["source"].isin(co.columns), "nodetype"] = "factor"
    edgelist.loc[edgelist["source"].isin(co.index), "nodetype"] = "population"
    # scale to -1:1
    edgelist["value"] = (edgelist["value"] - 0.5) * 2
    edgelist["absvalue"] = edgelist["value"].abs()
    edgelist.to_csv(prefix + f".{label}.network.csv", index=False)

    g = nx.from_pandas_edgelist(
        edgelist, edge_attr=["value", "edgetype", "absvalue"]
    )

    nx.set_node_attributes(
        g, {n: "factor" for n in co.columns}, name="nodetype"
    )
    nx.set_node_attributes(
        g, {n: "population" for n in edges["target"].unique()}, name="nodetype"
    )
    nx.write_gexf(g, prefix + f".{label}.network.gexf")


# Visualize severity state transition
_res = list()
for mdl in ["5a-", "2-", "5b-"]:
    model_name = [m for m in models if m.startswith(mdl)][0]
    model = models[model_name]
    prefix = f"differential.{model_name}.{reduction}."
    _res.append(
        pd.read_csv(
            output_dir / f"differential.{model_name}.{reduction}.results.csv"
        ).assign(model=model_name)
    )

res = pd.concat(_res)
res = res.loc[res["comparison"].str.startswith("severity_group")]

prefix = output_dir / "differential.all_models.coefficients"
cmap = plt.get_cmap("coolwarm")
norm = matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0)
for panel_name in panels:
    vars = panels[panel_name]

    res2 = res.loc[res["variable"].isin(vars)].copy()
    res2["sig"] = res2["qval"] < 0.05

    if panel_name == "Memory":
        res2 = res2.query("variable.str.contains('__').values", engine="python")

    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    sns.barplot(
        data=res2.groupby("comparison")["sig"].sum().reset_index(),
        x="comparison",
        y="sig",
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.savefig(prefix + f".{panel_name}.n_significant.svg", **figkws)
    plt.close(fig)

    # scale coefs to -2:2 just for the color
    res2["color"] = (res2["coef"] - -2) / (2 - -2)

    n = res2.shape[0] / 3
    fig, axes = plt.subplots(1, 3, figsize=(3 * 2.5, 0.15 * n), sharey=True)
    order = None
    for i, m in enumerate(res2["model"].unique()):
        axes[i].set(title=m, xlabel="log2(Fold-change) " + r"($\beta$)")
        axes[i].axvline(0, linestyle="--", color="grey")
        res3 = res2.loc[res2["model"] == m].set_index("variable")
        if order is None:
            res3 = res3.sort_values("coef")
            order = res3.index.tolist()
        else:
            res3 = res3.reindex(order)
        axes[i].scatter(res3["coef"], res3.index, color="grey", zorder=2)
        for j, x in res3.iterrows():
            axes[i].plot(
                (x["ci_0.025"], x["ci_0.975"]),
                (j, j),
                c=cmap(x["color"]),
                zorder=-1,
            )
            if x["qval"] <= 0.05:
                # axes[i].text(x['coef'], j, s="*")  # , "$_{*}$")
                axes[i].scatter(
                    x["coef"], j, marker="*", color="black", s=50, zorder=3
                )  # , "$_{*}$")
    fig.savefig(prefix + f".{panel_name}.change_severity_ci.svg", **figkws)
    plt.close(fig)


# Visualize treatment state transition
_res = list()
for mdl in ["2-", "3-"]:
    model_name = [m for m in models if m.startswith(mdl)][0]
    model = models[model_name]
    prefix = f"differential.{model_name}.{reduction}."
    _res.append(
        pd.read_csv(
            output_dir / f"differential.{model_name}.{reduction}.results.csv"
        ).assign(model=model_name)
    )

res = pd.concat(_res)
res = res.loc[res["comparison"].str.contains("severity_group|tocilizumab")]
q = res.pivot_table(columns="comparison", index="variable", values="coef")
diff = (
    (q.iloc[:, 1] - q.iloc[:, 0])
    .sort_values()
    .rename("diff")
    .to_frame()
    .join(q.mean(1).rename("mean"))
    .dropna()
)


# # Get largest coeficients only for toci treatment
top_n = 20
res2 = res.loc[res["comparison"].str.contains("tocilizumab")]
diff_sel = (
    res2.set_index("variable")["coef"].abs().sort_values().tail(top_n).index
)
res2 = res2.loc[res2["variable"].isin(diff_sel)]

# scale coefs to -2:2 just for the color
res2["color"] = (res2["coef"] - -2) / (2 - -2)

c = res2["comparison"].nunique()
n = res2.shape[0] / c
fig, ax = plt.subplots(1, c, figsize=(c * 2.0, 0.15 * n), sharey=True)
order = None
ax.set(title=m, xlabel="log2(Fold-change) " + r"($\beta$)")
ax.axvline(0, linestyle="--", color="grey")
res3 = res2.loc[res2["model"] == m].set_index("variable")
if order is None:
    res3 = res3.sort_values("coef")
    order = res3.index.tolist()
else:
    res3 = res3.reindex(order)
ax.scatter(res3["coef"], res3.index, color="grey")
for j, x in res3.iterrows():
    ax.plot((x["ci_0.025"], x["ci_0.975"]), (j, j), c=cmap(x["color"]))
fig.savefig(prefix + f".change_treatment_ci.top_{top_n}.svg", **figkws)
plt.close(fig)


# # Get 'discordant' changes
diff_sel = q.loc[
    (
        ((q.iloc[:, 1] < 0) & (q.iloc[:, 0] > 0))
        | ((q.iloc[:, 1] > 0) & (q.iloc[:, 0] < 0))
    )
    & (q.abs().mean(1) > 0.1)
].index
# diff_sel = diff.loc[(diff['mean'].abs() > 1) & (diff['diff'].abs() > 1)].index
res2 = res.loc[res["variable"].isin(diff_sel)]

# scale coefs to -2:2 just for the color
res2["color"] = (res2["coef"] - -2) / (2 - -2)

c = res["comparison"].nunique()
n = res2.shape[0] / c
fig, axes = plt.subplots(1, c, figsize=(c * 2.0, 0.15 * n), sharey=True)
order = None
for i, m in enumerate(res2["model"].unique()):
    axes[i].set(title=m, xlabel="log2(Fold-change) " + r"($\beta$)")
    axes[i].axvline(0, linestyle="--", color="grey")
    res3 = res2.loc[res2["model"] == m].set_index("variable")
    if order is None:
        res3 = res3.sort_values("coef")
        order = res3.index.tolist()
    else:
        res3 = res3.reindex(order)
    axes[i].scatter(res3["coef"], res3.index, color="grey")
    for j, x in res3.iterrows():
        axes[i].plot((x["ci_0.025"], x["ci_0.975"]), (j, j), c=cmap(x["color"]))
fig.savefig(prefix + ".change_treatment_ci.discordant.svg", **figkws)
plt.close(fig)


# Add lock file
open(output_dir / "__done__", "w")
