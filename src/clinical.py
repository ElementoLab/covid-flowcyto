#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

from src.conf import *


output_dir = results_dir / "clinical"
output_dir.mkdir(exist_ok=True, parents=True)


variables = CATEGORIES + CONTINUOUS


meta = pd.read_parquet(metadata_file)

remove = [
    "COVID19",
    "patient",
    "hyperlypidemia",
    "heme",
    "bone_marrow_transplant",
    "leukemia_lymphoma",
    "sleep_apnea",
    "pcr",
    "tocilizumab_pretreatment",
    "tocilizumab_postreatment",
    "processing_batch_categorical",
    "datesamples_continuous",
    "processing_batch_continuous",
]
variables = [v for v in variables if v not in remove]

# cats = meta.columns[meta.dtypes == pd.CategoricalDtype()]


# Variable correlation

to_corr = pd.get_dummies(
    meta.drop(remove, axis=1).drop_duplicates(subset="patient_code")[variables]
)
corrs = to_corr.corr(method="spearman")

kwargs = dict(
    center=0,
    cmap="RdBu_r",
    cbar_kws={"label": "Spearman correlation"},
    square=True,
)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(corrs, ax=ax, **kwargs)
fig.savefig(
    output_dir / "clinial_parameters.parameter_correlation.heatmap.svg",
    **figkws
)


# first do the correlation with nan filled
grid = sns.clustermap(
    corrs.fillna(np.nanmean(corrs.values)), metric="correlation", **kwargs
)
grid.savefig(
    output_dir / "clinial_parameters.parameter_correlation.clustermap.svg",
    **figkws
)

# then plot with heatmap ordered to still display NaNs
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(
    corrs.iloc[
        grid.dendrogram_row.reordered_ind, grid.dendrogram_col.reordered_ind
    ],
    ax=ax,
    **kwargs
)
fig.savefig(
    output_dir / "clinial_parameters.parameter_correlation.clustermap.svg",
    **figkws
)

# Patient correlation in clinical variables only
to_corr = pd.get_dummies(
    meta.query("severity_group != 'negative'")
    .drop(remove, axis=1)
    .drop_duplicates(subset="patient_code")[variables]
)
to_corr = to_corr.drop(to_corr.columns[(to_corr == 0).all()], axis=1)

f_corr = to_corr.corr(method="spearman")
p_corr = to_corr.T.corr(method="spearman")

grid = sns.clustermap(
    f_corr.corr(method="spearman"),
    center=0,
    cmap="RdBu_r",
    cbar_kws=dict(label="Spearman correlation"),
    xticklabels=True,
    yticklabels=True,
    figsize=(12, 11),
)
grid.savefig(
    output_dir / "clinial_parameters.parameter_correlation.patients.svg",
    **figkws
)


grid = sns.clustermap(
    p_corr.corr(method="spearman"),
    center=0,
    cmap="RdBu_r",
    cbar_kws=dict(label="Spearman correlation"),
    xticklabels=True,
    yticklabels=True,
    row_colors=sample_variables,
    colors_ratio=0.15 / sample_variables.shape[1],
    figsize=(12, 11),
)
grid.savefig(
    output_dir / "clinial_parameters.sample_correlation.patients.svg", **figkws
)


# Get Supl Fig. 1 / Table
# # Summary stats
# # Test for imbalances in clinical data between severity groups


# Add lock file
open(output_dir / "__done__", "w")
