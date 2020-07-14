#!/usr/bin/env python

"""
Visualizations of the cohort and the associated clinical data
"""

from src.conf import *


output_dir = results_dir / "clinical"
output_dir.mkdir(exist_ok=True, parents=True)


variables = CATEGORIES + CONTINUOUS


meta = pd.read_parquet(metadata_file)


# Variable correlation

to_corr = meta.drop_duplicates(subset="patient_code")[variables].copy()

for col in to_corr.columns[to_corr.dtypes == "category"]:
    to_corr[col] = meta[col].cat.codes

for col in to_corr.columns[
    to_corr.dtypes.apply(lambda x: x.name).str.contains("datetime")
]:
    to_corr[col] = minmax_scale(to_corr[col])

corrs = to_corr.corr(method="spearman").drop("patient", 0).drop("patient", 1)


kwargs = dict(
    # center=0,
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
grid = sns.clustermap(corrs.fillna(np.nanmean(corrs.values)), **kwargs)
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
p_corr = to_corr.query("COVID19 == 1").loc[:, corrs.index]
# # quickly impute missing batch for three patients
for col in p_corr.columns[p_corr.isnull().any()]:
    p_corr.loc[p_corr[col].isnull(), col] = p_corr[col].dropna().mean()
grid = sns.clustermap(
    p_corr.T.corr(method="spearman"),
    center=0.5,
    cmap="RdBu_r",
    cbar_kws=dict(label="Spearman correlation"),
    xticklabels=True,
    yticklabels=True,
    row_colors=sample_variables,
    colors_ratio=0.15 / sample_variables.shape[1],
    figsize=(12, 11),
)
grid.savefig(output_dir / "clinial_parameters.sample_correlation.svg", **figkws)


# Get Supl Fig. 1 / Table
# # Summary stats
# # Test for imbalances in clinical data between severity groups


# Add lock file
open(output_dir / "__done__", "w")
