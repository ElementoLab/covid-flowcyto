#!/usr/bin/env python

"""
This script does supervised analysis of the gated flow cytometry data.

The main analysis is the fitting of linear models on these data.

A few issues and a few options for each:
 - design:
     - controls were sampled one or more times while cases only once:
         - reduce controls by mean? -> can't model batch
         - add patient as mixed effect? -> don't have more than one sample for cases
 - missing data:
     - continuous:
         - imputation: only ~0.1% missing so, no brainer
     - categoricals:
         - drop
         - imputation?: circular argumentation - no go
 - proportion nature of the data:
     - z-score (loose sensitivity, ~harder to interpret coefficients)
     - logistic reg (did not converge for many cases :()
     - use Binomial GLM (no power?)
     - use Gamma GLM + log link (ok, but large coefficients sometimes :/)
     - use Gamma GLM + log link + regularization -> seems like the way to go
"""

import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from adjustText import adjust_text
import parmap

from imc.graphics import to_color_series

from src.conf import *


def add_colorbar(im, ax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation="vertical")


def rename_forward(x: Series) -> Series:
    return (
        x.str.replace("/", "___")
        .str.replace("+", "pos")
        .str.replace("-", "neg")
        .str.replace("(", "_O_")
        .str.replace(")", "_C_")
    )


def rename_back(x: Union[Series, str]) -> Union[Series, str]:
    if isinstance(x, str):
        _x = pd.Series(x)
    y = (
        _x.str.replace("pos", "+")
        .str.replace("neg", "-")
        .str.replace("_O_", "(")
        .str.replace("_C_", ")")
        .str.replace("___", "/")
    )
    return y[0] if isinstance(x, str) else y


def fit_model(variable, covariates, data, formula=None):
    cols = [
        "coef",
        "ci_0.025",
        "ci_0.975",
        "pval",
        "bse",
        "llf",
        "aic",
        "bic",
        "variable",
    ]
    if formula is None:
        formula = f"{variable} ~ {' + '.join(covariates)}"
    else:
        formula = variable + formula
    fam = sm.families.Gamma(sm.families.links.log())
    md = smf.glm(formula, data, family=fam)
    try:
        mdf = md.fit_regularized(
            maxiter=100, refit=True
        )  # , L1_wt=1 # <- Ridge
    except ValueError:  # this happens for variable: 'InegMDSC___All_CD45__O_WBC_C_'
        empty = pd.DataFrame(index=md.exog_names, columns=cols)
        print(f"Could not fit variable {variable}.")
        return empty
    params = pd.Series(mdf.params, index=md.exog_names, name="coef")
    conf_int = pd.DataFrame(
        mdf.conf_int(), index=params.index, columns=["ci_0.025", "ci_0.975"]
    )
    pvalues = pd.Series(mdf.pvalues, index=md.exog_names, name="pval")
    bse = pd.Series(mdf.bse, index=md.exog_names, name="bse")

    res = (
        params.to_frame()
        .assign(
            variable=rename_back(variable),
            llf=mdf.llf,
            aic=mdf.aic,
            bic=mdf.bic,
        )
        .join(conf_int)
        .join(pvalues)
        .join(bse)
    )
    return res


output_dir = results_dir / "supervised"
output_dir.mkdir(exist_ok=True, parents=True)

meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(matrix_imputed_file).sort_index(0).sort_index(1)
matrix_red_var = (
    pd.read_parquet(matrix_imputed_reduced_file).sort_index(0).sort_index(1)
)

alpha_thresh = 0.05
log_alpha_thresh = -np.log10(alpha_thresh)


# to annotate variables
cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns
parent_population = cols[1].rename("parent_population")


# Decide if it makes sense to use full matrix of with less variable redundancy
MATRIX_TO_USE = matrix_red_var
# in order to use the variable names in linear model, names must be cleaned up
matrix_c = MATRIX_TO_USE.copy()
matrix_c.columns = rename_forward(matrix_c.columns)

# Decide if using all samples (including technical replicates or reduced version)
# This is a reduced version, where replicates are averaged
meta_reduced = meta.drop_duplicates(subset=["sample_id"]).sort_values(
    "sample_id"
)
matrix_c_reduced = (
    matrix_c.groupby(meta["sample_id"]).mean().set_index(meta_reduced.index)
)


# Read up model specifications
specs = json.load(open("metadata/model_specifications.json", "r"))
models: Dict[str, Model] = dict()
for name, model in specs.items():
    models[name] = Model(**model)


results = dict()

# Fit
meta_red = pd.read_parquet(metadata_dir / "annotation.reduced_per_patient.pq")
red_pat_early = pd.read_parquet("data/matrix_imputed_reduced.red_pat_early.pq")
red_pat_early.columns = rename_forward(red_pat_early.columns)
red_pat_median = pd.read_parquet(
    "data/matrix_imputed_reduced.red_pat_median.pq"
)
red_pat_median.columns = rename_forward(red_pat_median.columns)


for model_name, model in models.items():
    formula = model["formula"] if "formula" in model else None

    ## m, d, label, fit_vars = (meta_reduced, matrix_c_reduced, "reduced", variables)
    matrices = [
        (meta, matrix_c, "original", model["covariates"]),
        (meta_reduced, matrix_c_reduced, "reduced", model["covariates"]),
        (meta_red, red_pat_early, "reduced_early", model["covariates"],),
        (meta_red, red_pat_median, "reduced_median", model["covariates"],),
    ]
    for m, d, label, fit_vars in matrices:
        # data = zscore(d).join(m[fit_vars]).dropna()
        data = d.join(m[fit_vars]).dropna()

        # remove unused levels
        for cat in data.columns[data.dtypes == "category"]:
            data[cat] = data[cat].cat.remove_unused_categories()

        u = data.nunique() == 1
        if u.any():
            print(f"'{', '.join(data.columns[u])}' have only one value.")
            print("Removing from model.")
            fit_vars = [v for v in fit_vars if v not in data.columns[u]]
            data = data.drop(data.columns[u], axis=1)

        # Keep record of exactly what was the input to the model:
        data.sort_values(fit_vars).to_csv(
            output_dir / f"model_X_matrix.{model_name}.{label}.csv"
        )
        _res = parmap.map(
            fit_model,
            d.columns,
            covariates=fit_vars,
            data=data,
            formula=formula,
            pm_pbar=True,
        )
        res = pd.concat(_res).rename_axis(index="comparison")
        res["qval"] = multipletests(res["pval"].fillna(1), method="fdr_bh")[1]
        res["log_pval"] = log_pvalues(res["pval"]).fillna(0)
        res["log_qval"] = log_pvalues(res["qval"]).fillna(0)
        res.to_csv(
            output_dir / f"differential.{model_name}.{label}.results.csv"
        )
        results[(model_name, label)] = res


# Compare models (original vs reduced data)
from scipy.stats import pearsonr

opts = [
    ("original", "reduced"),
    ("reduced", "reduced_early"),
    ("reduced", "reduced_median"),
]

for a, b in opts:
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(3 * len(models), 3 * 2),
        # sharex="row",
        # sharey="row",
    )
    for i, (model_name, _) in enumerate(models.items()):
        x = results[(model_name, a)].drop("Intercept")
        y = results[(model_name, b)].drop("Intercept")

        assert (x.index == y.index).all()
        close = np.allclose(x["coef"], y["coef"])
        kw = dict(linestyle="--", color="grey", alpha=0.2)
        kws = dict(s=2, alpha=0.5, rasterized=True)

        cx = x["coef"].clip(-20, 20)
        cy = y["coef"].clip(-20, 20)
        c = pd.concat([cx, cy], 1)
        cdna = c.dropna()
        lpx = log_pvalues(x["pval"])  # , clip=0.95
        lpy = log_pvalues(y["pval"])  # , clip=0.95
        lp = pd.concat([lpx, lpy], 1)
        lpdna = lp.dropna()

        # axes[0, i].scatter(np.tanh(x["coef"]), np.tanh(y["coef"]), **kws)
        cv = max(cx.abs().max(), cy.abs().max())
        mcv = -((1 / 8) * cv)
        pv = max(lpx.max(), lpy.max())
        cv += cv * 0.1
        pv += pv * 0.1

        axes[0, i].plot((-cv, cv), (-cv, cv), **kw)
        axes[1, i].plot((mcv, pv), (mcv, pv), **kw)

        axes[0, i].scatter(cx, cy, c=lp.mean(1), **kws)
        axes[1, i].scatter(lpx, lpy, c=c.abs().mean(1), **kws)
        r, p = pearsonr(cdna.iloc[:, 0], cdna.iloc[:, 1])
        axes[0, i].set(
            title=model_name + f"\nr = {r:.3f}, p = {p:.2e}",
            xlim=(-cv, cv),
            ylim=(-cv, cv),
        )
        r, p = pearsonr(lpdna.iloc[:, 0], lpdna.iloc[:, 1])
        axes[1, i].set(
            title=f"r = {r:.3f}, p = {p:.2e}",
            xlabel=a.capitalize(),
            xlim=(mcv, pv),
            ylim=(mcv, pv),
        )
    axes[0, 0].set(ylabel=f"Coefficient\n{b.capitalize()}")
    axes[1, 0].set(ylabel=f"-log10(P-value)\n{b.capitalize()}")
    fig.savefig(
        output_dir / f"differential.model_comparison.{a}_vs_{b}.svg", **figkws
    )
    plt.close(fig)


label = "reduced"
matrix_c = matrix_red_var
for i, (model_name, model) in enumerate(list(models.items())[1:]):
    # break
    prefix = f"differential.{model_name}.{label}."
    res = pd.read_csv(
        output_dir / f"differential.{model_name}.{label}.results.csv"
    )
    res = res.loc[res["llf"] < np.inf]

    long_f = res.pivot_table(index="variable", columns="comparison")
    # drop columns with levels not estimated
    long_f = long_f.loc[
        :,
        long_f.columns.get_level_values(1).isin(
            long_f["coef"].columns[long_f["coef"].abs().sum() > 1e-10]
        ),
    ]

    coefs = long_f["coef"]

    for variable in model["continuous"]:
        coefs[variable] = coefs[variable] * (meta[variable].mean() / 2)
    pvals = long_f["pval"]
    qvals = long_f["qval"]
    lpvals = long_f["log_pval"]
    lqvals = long_f["log_qval"]

    # Visualize

    # # Heatmaps
    ks = dict(center=0, cmap="RdBu_r", robust=True, metric="correlation")
    grid = sns.clustermap(coefs, cbar_kws=dict(label="log2(fold-change)"), **ks)
    grid.savefig(output_dir / prefix + "lfc.all_vars.clustermap.svg")
    grid = sns.clustermap(lpvals, cbar_kws=dict(label="-log10(p-value)"), **ks)
    grid.savefig(output_dir / prefix + "pvals_only.all_vars.clustermap.svg")

    # # # Heatmap combining both change and significance
    cols = ~coefs.columns.str.contains("|".join(technical))
    grid = sns.clustermap(
        coefs.loc[:, cols].drop("Intercept", axis=1),
        cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
        row_colors=lpvals.loc[:, cols],
        xticklabels=True,
        yticklabels=True,
        **ks,
    )
    grid.savefig(output_dir / prefix + "join_lfc_pvals.all_vars.clustermap.svg")

    # # # only significatnt
    sigs = (lqvals >= log_alpha_thresh).any(1)
    grid = sns.clustermap(
        coefs.loc[sigs, cols].drop("Intercept", axis=1),
        cbar_kws=dict(label="log2(Fold-change) " + r"($\beta$)"),
        row_colors=lpvals.loc[sigs, cols],
        xticklabels=True,
        yticklabels=True,
        **ks,
    )
    grid.savefig(
        output_dir / prefix
        + f"join_lfc_pvals.p<{alpha_thresh}_only.clustermap.svg"
    )

    # # Volcano plots
    for variable in model["covariates"]:
        cols = lpvals.columns[lpvals.columns.str.contains(variable)]
        n = len(cols)
        if not n:
            continue

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
        axes = axes.squeeze(0)
        for i, col in enumerate(cols):
            sigs = lqvals[col] >= log_alpha_thresh
            kwargs = dict(s=2, alpha=0.5, color="grey")
            axes[i].scatter(
                coefs.loc[~sigs, col], lpvals.loc[~sigs, col], **kwargs
            )
            kwargs = dict(
                s=10, alpha=1.0, c=lqvals.loc[sigs, col], cmap="Reds", vmin=0
            )
            im = axes[i].scatter(
                coefs.loc[sigs, col], lpvals.loc[sigs, col], **kwargs
            )
            if "[" in col:
                name = re.findall(r"^(.*)\[", col)[0]
                inst = re.findall(r"\[T.(.*)\]", col)[0]
            else:
                name = inst = col

            # v = -np.log10(multipletests([alpha_thresh] * coefs[col].shape[0])[1][0])
            v = lpvals.loc[~sigs, col].max()
            axes[i].axhline(v, color="grey", linestyle="--", alpha=0.5)
            axes[i].axvline(0, color="grey", linestyle="--", alpha=0.5)
            try:
                axes[i].set_title(f"{name}: {inst}/{meta[variable].min()}")
            except TypeError:
                axes[i].set_title(f"{name}: {inst}")
            axes[i].set_xlabel("log2(Fold-change) " + r"($\beta$)")
            axes[i].set_ylabel("-log10(p-value)")

            texts = text(
                coefs.loc[sigs, col],
                lpvals.loc[sigs, col],
                coefs.loc[sigs, col].index,
                axes[i],
                fontsize=5,
            )
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="->", color="black"),
                ax=axes[i],
            )
            add_colorbar(im, axes[i])

        fig.savefig(
            output_dir / prefix + f"test_{variable}.volcano.svg", **figkws
        )
        plt.close(fig)

    # # MA plots
    for variable in model["covariates"]:
        cols = lpvals.columns[lpvals.columns.str.contains(variable)]
        n = len(cols)
        if not n:
            continue

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
        axes = axes.squeeze(0)
        for i, col in enumerate(cols):
            sigs = lqvals[col] >= log_alpha_thresh
            kwargs = dict(s=2, alpha=0.5, color="grey")
            axes[i].scatter(
                coefs.loc[~sigs, "Intercept"], coefs.loc[~sigs, col], **kwargs,
            )
            kwargs = dict(
                s=10, alpha=1.0, c=lqvals.loc[sigs, col], cmap="Reds", vmin=0
            )
            im = axes[i].scatter(
                coefs.loc[sigs, "Intercept"], coefs.loc[sigs, col], **kwargs
            )
            if "[" in col:
                name = re.findall(r"^(.*)\[", col)[0]
                inst = re.findall(r"\[T.(.*)\]", col)[0]
            else:
                name = inst = col
            axes[i].axhline(0, color="grey", linestyle="--", alpha=0.5)
            axes[i].set_title(f"{name}: {inst}/{meta[variable].min()}")
            axes[i].set_xlabel("Mean")
            axes[i].set_ylabel("log2(fold-change)")

            texts = text(
                coefs.loc[sigs, "Intercept"],
                coefs.loc[sigs, col],
                coefs.loc[sigs, col].index,
                axes[i],
                fontsize=5,
            )
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="->", color="black"),
                ax=axes[i],
            )
            add_colorbar(im, axes[i])

        fig.savefig(
            output_dir / prefix + f"test_{variable}.maplots.svg", **figkws
        )
        plt.close(fig)

    # # Illustration of top hits
    n_plot = 1000
    patches = (
        matplotlib.collections.PatchCollection,
        matplotlib.collections.PathCollection,
    )
    for variable in model["categories"]:  # "variables"
        cols = pvals.columns[pvals.columns.str.contains(variable)]
        control = meta[variable].min()
        n = len(cols)
        if not n:
            continue
        for i, col in enumerate(cols):
            v = lqvals[col].sort_values()
            sigs = v  # v[v >= log_alpha_thresh]
            if sigs.empty:
                continue
            # print(variable, sigs)
            sigs = sigs.tail(n_plot).index[::-1]
            data = matrix_c[sigs].join(meta[variable]).melt(id_vars=variable)
            # for the interaction models ->
            # data = matrix_c[sigs].join(meta[['severity_group',  variable]]).query("severity_group == 'severe'").drop("severity_group", 1).melt(id_vars=variable)

            kws = dict(
                data=data, x=variable, y="value", hue=variable, palette="tab10"
            )
            grid = sns.FacetGrid(
                data=data, col="variable", sharey=False, height=3, col_wrap=4
            )
            if variable in model["categories"]:
                grid.map_dataframe(
                    sns.boxenplot, saturation=0.5, dodge=False, **kws
                )
            else:
                grid.map_dataframe(sns.scatterplot, **kws)

            for ax in grid.axes.flat:
                [
                    x.set_alpha(0.25)
                    for x in ax.get_children()
                    if isinstance(x, patches)
                ]
            grid.map_dataframe(sns.swarmplot, **kws)

            for ax in grid.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

            # add stats to title
            group = re.findall(r"^.*\[T.(.*)\]", col)[0] if "[" in col else col
            for ax in grid.axes.flat:
                var = ax.get_title().replace("variable = ", "")
                pop = var
                try:
                    pop, parent = re.findall(r"(.*)/(.*)", pop)[0]
                    ax.set_ylabel(f"% {parent}")
                except IndexError:
                    pass
                ax.set_title(
                    pop
                    + f"\n{group}/{control}:\n"
                    + f"Coef = {coefs.loc[var, col]:.3f}; "
                    + f"FDR = {qvals.loc[var, col]:.3e}"
                )

            # grid.map(sns.boxplot)
            grid.savefig(
                output_dir / prefix + f"test_{variable}.{col}.swarm.svg"
            )
            plt.close(grid.fig)


# Add lock file
open(output_dir / "__done__", "w")


# # Plot volcano for additional contrasts

# model_name = "categoricals"
# cur_variables = categories + technical

# m, d, label, fit_vars = (meta, matrix_c, "original", cur_variables)
# # data = zscore(d).join(m[fit_vars]).dropna()
# data = d.join(m[fit_vars]).dropna()

# base = "mild"
# data["severity_group"] = data["severity_group"].cat.reorder_categories(
#     ["mild", "negative", "non-covid", "severe", "convalescent"]
# )
# # base = "severe"
# # data["severity_group"] = data["severity_group"].cat.reorder_categories(["severe", "negative", "non-covid", "mild", "convalescent"])
# _res = list()
# for col in tqdm(d.columns):
#     # data[col] = data[col] / 100  # for logit or binomial

#     formula = f"{col} ~ {' + '.join(fit_vars)}"
#     # formula = f"{col} ~ severity_group"
#     # md = smf.glm(formula, data)
#     md = smf.glm(
#         formula, data, family=sm.families.Gamma(sm.families.links.log())
#     )
#     # md = smf.logit(formula, data)
#     # md = smf.glm(formula, data, family=sm.families.Binomial())

#     mdf = md.fit(maxiter=100)
#     # mdf = md.fit_regularized(maxiter=100, refit=True)  # , L1_wt=1 # <- Ridge
#     params = pd.Series(mdf.params, index=md.exog_names, name="coef")
#     pvalues = pd.Series(mdf.pvalues, index=md.exog_names, name="pval")

#     # fig, ax = plt.subplots()
#     # sns.boxplot(data=data, x="severity_group", y=col, ax=ax)
#     # sns.swarmplot(data=data, x="severity_group", y=col, ax=ax)
#     # ax.set_title(str(params[params.index.str.contains("severity_group")]))

#     _res.append(
#         params.to_frame().join(pvalues).assign(variable=rename_back(col))
#     )

# r = pd.concat(_res).rename_axis(index="comparison")

# category = "severity_group"

# long_f = r.loc[r.index.str.contains(category)].pivot_table(
#     index="variable", columns="comparison"
# )

# long_f.index = matrix.columns

# coefs = long_f["coef"]
# pvals = long_f["pval"]
# lpvals = log_pvalues(pvals).fillna(0)
# qvals = (
#     long_f["pval"]
#     .apply(multipletests, method="fdr_bh")
#     .apply(lambda x: pd.Series(x[1]))
#     .T.set_index(long_f.index)
# )
# lqvals = log_pvalues(qvals)

# # Visualize

# # # Volcano plots

# cols = lpvals.columns[lpvals.columns.str.contains(category)]
# cols = cols[~cols.str.contains("covid|negative")]
# n = len(cols)
# fig, axes = plt.subplots(1, n, figsize=(n * 4, 1 * 4), squeeze=False)
# axes = axes.squeeze(0)
# for i, col in enumerate(cols):
#     sigs = lqvals[col] >= log_alpha_thresh
#     kwargs = dict(s=2, alpha=0.5, color="grey")
#     axes[i].scatter(coefs.loc[~sigs, col], lpvals.loc[~sigs, col], **kwargs)
#     kwargs = dict(s=10, alpha=1.0, c=lqvals.loc[sigs, col], cmap="Reds", vmin=0)
#     im = axes[i].scatter(coefs.loc[sigs, col], lpvals.loc[sigs, col], **kwargs)
#     name = re.findall(r"^(.*)\[", col)[0]
#     inst = re.findall(r"\[T.(.*)\]", col)[0]
#     # v = -np.log10(multipletests([alpha_thresh] * coefs[col].shape[0])[1][0])
#     v = lpvals.loc[~sigs, col].max()
#     axes[i].axhline(v, color="grey", linestyle="--", alpha=0.5)
#     axes[i].axvline(0, color="grey", linestyle="--", alpha=0.5)
#     axes[i].set_title(f"{name}: {inst}/{data[category].min()}")
#     axes[i].set_xlabel("log2(Fold-change) " + r"($\beta$)")
#     axes[i].set_ylabel("-log10(p-value)")

#     texts = text(
#         coefs.loc[sigs, col],
#         lpvals.loc[sigs, col],
#         coefs.loc[sigs, col].index,
#         axes[i],
#         fontsize=5,
#     )
#     adjust_text(
#         texts, arrowprops=dict(arrowstyle="->", color="black"), ax=axes[i]
#     )
#     add_colorbar(im, axes[i])

# fig.savefig(
#     output_dir / f"differential.{label}.test_{category}.volcano.over_{base}.svg"
# )
# plt.close(fig)


# # Illustration of top hits
n_plot = 20
patches = (
    matplotlib.collections.PatchCollection,
    matplotlib.collections.PathCollection,
)

interactions = ["severity_group", "intubation", "death", "hospitalization"]

for model_name, model in {
    k: v for k, v in models.items() if "interaction" in k
}.items():
    res = pd.read_csv(
        output_dir / f"differential.{model_name}.{label}.results.csv",
        index_col=0,
    )
    prefix = f"differential.{model_name}.{label}.interaction_sex."
    r = res.sort_values("pval")
    r = r.loc[r.index.str.contains(":")]
    r = r.loc[r["coef"].abs() < 6]
    sigs = r.head(n_plot)["variable"]

    variable = [x for x in model["covariates"] if x != "sex"][0]
    d = (
        meta[[variable, "sex"]]
        .join(matrix[sigs])
        .melt(id_vars=[variable, "sex"])
        .dropna()
    )

    for cat in d.columns[d.dtypes == "category"]:
        d[cat] = d[cat].cat.remove_unused_categories()
    # grid = sns.catplot(data=d, x=variable, y='value', col='variable', col_wrap=4, hue="sex", sharey=False)

    kws = dict(data=d, x=variable, y="value", hue="sex", palette="tab10",)
    grid = sns.FacetGrid(
        data=d, col="variable", sharey=False, height=3, col_wrap=4
    )
    grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=True, **kws)

    for ax in grid.axes.flat:
        [x.set_alpha(0.25) for x in ax.get_children() if isinstance(x, patches)]
    grid.map_dataframe(sns.swarmplot, dodge=True, **kws)

    for ax in grid.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # add stats to title
    # group = re.findall(r"^.*\[T.(.*)\]", col)[0] if "[" in col else col
    for ax in grid.axes.flat:
        var = ax.get_title().replace("variable = ", "")
        pop = var
        try:
            pop, parent = re.findall(r"(.*)/(.*)", pop)[0]
            ax.set_ylabel(f"% {parent}")
        except IndexError:
            pass

        s = res.loc[res["variable"] == var].drop(["Intercept"])
        s = s.loc[s.index.str.contains(":")]
        m = s["coef"].abs().argmax()
        pos = s.iloc[m].name
        c = s.iloc[m]["coef"]
        control = meta[variable].min()
        ax.set_title(
            pop
            + f"\n{pos}:\n"
            + f"Coef = {c:.3f}; "
            + f"FDR = {s['qval'].min():.3e}"
        )

    # grid.map(sns.boxplot)
    grid.savefig(output_dir / prefix + f"{variable}.swarm+boxenplot.svg")
    plt.close(grid.fig)
