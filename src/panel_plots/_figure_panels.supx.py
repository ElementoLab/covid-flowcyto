#!/usr/bin/env python

"""

"""

import re

import pingouin as pg

from src.conf import *


output_dir = figures_dir / "panels"
output_dir.mkdir()

myelo = matrix.columns[
    matrix.columns.str.contains("MDSC/All_CD45_(PBMC)", regex=False)
]

fig, axes = plt.subplots(3, 2, figsize=(7, 10), tight_layout=True)
for ax, pop in zip(axes, myelo):
    for cbc, name, ax in zip(
        ["lymph_CBC", "neutrophils"], ["Lymphocytes", "Neutrophils"], ax
    ):
        p = meta[[cbc]].join(matrix[pop]).dropna()
        sns.regplot(p[cbc], p[pop], scatter_kws=dict(s=2, alpha=0.5), ax=ax)
        res = pg.corr(p[cbc], p[pop], method="spearman").squeeze()
        f = np.array([0.1, 1.1])
        ax.set(
            title=f"r = {res['r']:.2f}; ci = {res['CI95%']}; p = {res['p-val']:.2e}",
            xlabel=f"{name} (%, Sysmex CBC)",
            ylabel=pop,
            # xlim=(-10, 110),
            xlim=np.asarray(ax.get_xlim()) * f,
            # ylim=(-10, 110),
            ylim=np.asarray(ax.get_ylim()) * f,
        )
fig.savefig(output_dir / "sysmex.neutrophil_lymphocyte.svg", **figkws)

lr = pg.linear_regression(p[pop], p["neutrophils"])
