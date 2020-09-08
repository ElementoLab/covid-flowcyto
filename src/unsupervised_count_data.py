import re

from sklearn.decomposition import PCA, NMF  # type: ignore
from sklearn.manifold import MDS, TSNE  # type: ignore
from umap import UMAP  # type: ignore

from imc.graphics import to_color_series

from src.conf import *

output_dir = results_dir / "unsupervised"

meta = pd.read_parquet(metadata_file)
matrix = pd.read_parquet(data_dir / "matrix.counts.pq")
matrix += 1
meta = meta.loc[matrix.index]

matrix_counts_median = matrix.groupby(meta["patient_code"]).median()

cols = matrix.columns.str.extract("(.*)/(.*)")
cols.index = matrix.columns


meta_red = pd.read_parquet(metadata_dir / "annotation.reduced_per_patient.pq")
red_pat_early = pd.read_parquet("data/matrix_imputed_reduced.red_pat_early.pq")
red_pat_median = pd.read_parquet(
    "data/matrix_imputed_reduced.red_pat_median.pq"
)


# Demonstrate the data
meta_c, matrix_c, reduction = (meta_red, matrix_counts_median, "reduced_median")

parent_population = cols[1].rename("parent_population")
variable_classes = parent_population.to_frame().join(
    matrix.mean().rename("Mean")
)

# # Plot counts ouL)major populations for each patient group
# + a few ratios like CD4/CD8 (of CD3+)
for cat_var in categories:
    # cat_var = "severity_group"
    for parent in variable_classes["parent_population"].unique():
        figfile = (
            output_dir
            / f"variable_illustration.counts.{cat_var}.panel_{parent}.swarm+boxen.svg"
        )
        if figfile.exists():
            continue

        data = (
            matrix_c.loc[
                :,
                variable_classes.query(
                    f"parent_population == '{parent}'"
                ).index,
            ]
            .join(meta_c[[cat_var]])
            .melt(
                id_vars=[cat_var],
                var_name="population",
                value_name="counts (uL)",
            )
        )

        kws = dict(
            data=data, x=cat_var, y="counts (uL)", hue=cat_var, palette="tab10",
        )
        grid = sns.FacetGrid(
            data=data, col="population", sharey=False, height=3, col_wrap=4
        )
        grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=False, **kws)
        # grid.map_dataframe(sns.stripplot, y="value", x=category, hue=category, data=data, palette='tab10')

        for ax in grid.axes.flat:
            for x in ax.get_children():
                if isinstance(x, patches):
                    x.set_alpha(0.25)
        grid.map_dataframe(sns.swarmplot, **kws)

        for ax in grid.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # add stats to title
        for ax in grid.axes.flat:
            ax.set(yscale="symlog")
            ax.set_ylim(bottom=0)
            var = ax.get_title().replace("population = ", "")
            try:
                child, parent = re.findall(r"(.*)/(.*)", var)[0]
                ax.set_title(child)
                ax.set_ylabel("Cells / uL")
            except IndexError:
                ax.set_title(var)

        # grid.map(sns.boxplot)
        grid.savefig(figfile)
        plt.close(grid.fig)


for cat_var in categories:
    if cat_var in ["sex", "processing_batch_categorical"]:
        continue
    for parent in variable_classes["parent_population"].unique():
        figfile = (
            output_dir
            / f"variable_illustration.counts.{cat_var}_interaction_sex.panel_{parent}.{reduction}.swarm+boxen.svg"
        )
        data = (
            matrix_c.loc[
                :,
                variable_classes.query(
                    f"parent_population == '{parent}'"
                ).index,
            ]
            .join(meta_c[[cat_var, "sex"]])
            .melt(
                id_vars=[cat_var, "sex"],
                var_name="population",
                value_name="counts (uL)",
            )
        )

        kws = dict(
            data=data, x=cat_var, y="counts (uL)", hue="sex", palette="tab10",
        )
        grid = sns.FacetGrid(
            data=data, col="population", sharey=False, height=3, col_wrap=4
        )
        grid.map_dataframe(sns.boxenplot, saturation=0.5, dodge=True, **kws)
        # grid.map_dataframe(sns.stripplot, y="value", x=category, hue=category, data=data, palette='tab10')

        for ax in grid.axes.flat:
            for x in ax.get_children():
                if isinstance(x, patches):
                    x.set_alpha(0.25)
        grid.map_dataframe(sns.swarmplot, dodge=True, **kws)

        for ax in grid.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # add stats to title
        for ax in grid.axes.flat:
            ax.set(yscale="symlog")
            ax.set_ylim(bottom=0)
            var = ax.get_title().replace("population = ", "")
            try:
                child, parent = re.findall(r"(.*)/(.*)", var)[0]
                ax.set_title(child)
                ax.set_ylabel("Cells / uL")
            except IndexError:
                ax.set_title(var)

        grid.savefig(figfile)
        plt.close(grid.fig)


import pingouin as pg

m = matrix.join(meta[["severity_group"]])
m["severity_group"] = m["severity_group"].cat.remove_unused_categories()
res = pd.concat(
    [
        pg.pairwise_ttests(
            data=m, dv=var, between="severity_group", parametric=False
        ).assign(variable=var)
        for var in m.columns[:-1]
    ]
).drop(["Contrast"], axis=1)
res["p-cor"] = pg.multicomp(res["p-unc"].values, method="fdr_bh")[1]
res.to_csv("diff.absolute.csv", index=False)
