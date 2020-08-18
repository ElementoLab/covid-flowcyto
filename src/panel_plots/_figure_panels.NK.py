import scanpy as sc
from mpl_toolkits.mplot3d import Axes3D

from src.conf import *

panel_name = "WB_NK_KIR"
label = "full"

panel_dir = results_dir / "single_cell" / panel_name
processed_h5ad = panel_dir / f"{panel_name}.concatenated.{label}.processed.h5ad"

a = sc.read(processed_h5ad)
x = a.to_df().join(a.obs)


cd158a = "CD158a(PerCP-Cy5-5-A)"
cd158i = "CD158i(APC-A)"
cd158b = "CD158b(PE-A)"
cd158e = "CD158e(FITC-A)"


fig = plt.figure(figsize=(4 * 3, 3))
for i, lab in enumerate(meta["severity_group"].cat.categories, 1):
    ax = fig.add_subplot(1, 4, i, projection="3d")
    y = x.query(f"severity_group == '{lab}'")
    points = ax.scatter(
        y[cd158i],
        y[cd158a],
        y[cd158b],
        c=y[cd158e],
        s=1,
        alpha=0.01,
        rasterized=True,
    )
    ax.set(xlabel="CD158i", ylabel="CD158a", zlabel="CD158b", title=lab)
    fig.colorbar(points, label=cd158e)
fig.savefig(
    figures_dir / "panels" / f"NKcells.3dplot.scatter.svg", **figkws,
)
