#!/usr/bin/env python

"""
This script plots the model estimates jointly for all.
"""

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


# # Make version with coefficients reduce by mean if appearing in more than one model
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
        cmap="binary",
        cbar_pos=None,
    )
    grid3.savefig(prefix + f".{label}.clustermap.sig_only.svg")
    plt.close(grid3.fig)


# Add lock file
open(output_dir / "__done__", "w")
