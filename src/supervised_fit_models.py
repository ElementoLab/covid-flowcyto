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

import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore
import parmap  # type: ignore

from src.conf import *


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

    return (
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

# Decide if using all samples (including technical replicates or reduced version)
# This is a reduced version, where replicates are averaged
meta_reduced = meta.drop_duplicates(subset=["sample_id"]).sort_values(
    "sample_id"
)
matrix_reduced = (
    matrix_red_var.groupby(meta["sample_id"])
    .mean()
    .set_index(meta_reduced.index)
)

# Read up various matrices  used for fitting
meta_red = pd.read_parquet(metadata_dir / "annotation.reduced_per_patient.pq")
red_pat_early = pd.read_parquet("data/matrix_imputed_reduced.red_pat_early.pq")
red_pat_median = pd.read_parquet(
    "data/matrix_imputed_reduced.red_pat_median.pq"
)

# Fit
for model_name, model in list(models.items()):
    formula = model["formula"] if "formula" in model else None

    ## m, d, reduction, covariates = (meta_reduced, matrix_reduced, "reduced", model['covariates'])
    matrices = [
        # (meta, matrix_red_var, "original", model["covariates"]),
        (meta_reduced, matrix_reduced, "reduced", model["covariates"],),
        # (meta_red, red_pat_early, "reduced_early", model["covariates"],),
        # (meta_red, red_pat_median, "reduced_median", model["covariates"],),
    ]
    for m, d, reduction, covariates in matrices:
        results_file = (
            output_dir / f"differential.{model_name}.{reduction}.results.csv"
        )
        # if results_file.exists():
        #     continue
        print(model_name, reduction)
        d.columns = rename_forward(d.columns)
        # data = zscore(d).join(m[covariates]).dropna()
        data = d.join(m[covariates]).dropna()

        # remove unused levels, ensuring the 'lowest' is the one to compare to
        for cat in data.columns[data.dtypes == "category"]:
            data[cat] = data[cat].cat.remove_unused_categories()

        u = data.nunique() == 1
        if u.any():
            print(f"'{', '.join(data.columns[u])}' have only one value.")
            print("Removing from model.")
            covariates = [v for v in covariates if v not in data.columns[u]]
            data = data.drop(data.columns[u], axis=1)

        # Keep record of exactly what was the input to the model:
        data.sort_values(covariates).to_csv(
            output_dir / f"model_X_matrix.{model_name}.{reduction}.csv"
        )
        _res = parmap.map(
            fit_model,
            d.columns,
            covariates=covariates,
            data=data,
            formula=formula,
            pm_pbar=True,
        )
        res = pd.concat(_res).rename_axis(index="comparison")
        res["qval"] = multipletests(res["pval"].fillna(1), method="fdr_bh")[1]
        res["log_pval"] = log_pvalues(res["pval"]).fillna(0)
        res["log_qval"] = log_pvalues(res["qval"]).fillna(0)
        res.to_csv(results_file)
