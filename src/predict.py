#!/usr/bin/env python

"""
Develop a classifier to distinguish between patients with different degrees of
disease severity.
"""

from sklearn.pipeline import Pipeline  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.svm import LinearSVC  # type: ignore[import]
from sklearn.model_selection import cross_validate  # type: ignore[import]

from joblib import Parallel, delayed  # type: ignore[import]

from src.conf import *


def fit(_, model, X, y, k=None):
    feature_attr = {
        RandomForestClassifier: "feature_importances_",
        LogisticRegression: "coef_",
        # ElasticNet: "coef_",
        LinearSVC: "coef_",
    }
    kws = dict(
        cv=10, scoring="roc_auc", return_train_score=True, return_estimator=True
    )

    # # Randomize order of both X and y (jointly)
    # X = X.sample(frac=1.0).copy()
    # y = y.reindex(X.index).copy()

    clf = model()

    # Build pipeline
    components = list()
    # # Z-score if needed
    if not isinstance(clf, RandomForestClassifier):
        components += [("scaler", StandardScaler())]
    # # Do feature selection if requested
    if k is not None:
        components += [("selector", SelectKBest(mutual_info_classif, k=k))]
    # # Finally add the classifier
    components += [("classifier", clf)]
    pipe = Pipeline(components)

    # Train/cross-validate with real data
    out1 = cross_validate(pipe, X, y, **kws)
    # Train/cross-validate with shuffled labels
    out2 = cross_validate(pipe, X, y.sample(frac=1.0), **kws)

    # Extract coefficients/feature importances
    feat = feature_attr[clf.__class__]
    coefs = tuple()
    for out in [out1, out2]:
        co = pd.DataFrame(
            [
                pd.Series(
                    getattr(c["classifier"], feat),
                    index=X.columns[c["selector"].get_support()]
                    if k is not None
                    else X.columns,
                )
                for c in out["estimator"]
            ]
        )
        # If keeping all variables, simply report mean
        if k is None:
            coefs += (co.mean(0),)
        # otherwise, simply count how often variable was chosen at all
        else:
            coefs += ((~co.isnull()).sum(),)
    return (
        out1["train_score"].mean(),
        out1["test_score"].mean(),
        out2["train_score"].mean(),
        out2["test_score"].mean(),
    ) + coefs


def predict_patient(_, clf, X, y, X_test, patient):
    # remove patient from training
    # fit on remaining patients
    clf = clf.fit(
        X.loc[~X.index.isin([patient])], y.loc[~y.index.isin([patient])]
    )
    # predict patient, return "severe" probability
    return clf.predict_log_proba(X_test)[:, 1]


output_dir = results_dir / "predict"
output_dir.mkdir()

# Read in dataframes reduced per patient (earliest timepoint)
meta_red = pd.read_parquet(metadata_dir / "annotation.reduced_per_patient.pq")
red_pat_early = pd.read_parquet("data/matrix_imputed_reduced.red_pat_early.pq")

# Use only mild-severe patients
m = meta_red.query(
    "severity_group.isin(['mild', 'severe']).values", engine="python"
)

# Convert classes to binary
y = m["severity_group"].cat.remove_unused_categories().cat.codes

# Align dataframes
X = red_pat_early.loc[m.index]
# # For the other classifiers

N = 1000

insts = [
    RandomForestClassifier,
    LogisticRegression,
    LinearSVC,
    # ElasticNet,
]
for label, k in [("", None), (".feature_selection", 8)]:
    for model in insts:
        name = str(type(model())).split(".")[-1][:-2]
        print(name)

        # Fit
        res = Parallel(n_jobs=-1)(
            delayed(fit)(i, model=RandomForestClassifier, X=X, y=y, k=k)
            for i in range(N)
        )
        # Get ROC_AUC scores only
        scores = pd.DataFrame(
            np.asarray([r[:-2] for r in res]),
            columns=[
                "train_score",
                "test_score",
                "train_score_random",
                "test_score_random",
            ],
        )
        scores.to_csv(
            output_dir / f"severe-mild_prediction.{name}{label}.scores.csv"
        )
        scores = pd.read_csv(
            output_dir / f"severe-mild_prediction.{name}{label}.scores.csv",
            index_col=0,
        )

        p = scores.loc[:, scores.columns.str.contains("test")].melt()

        if k is not None:
            fig, ax = plt.subplots(1, 1, figsize=(2, 4))
            ax.axhline(0.5, linestyle="--", color="grey")
            sns.boxenplot(data=p, x="variable", y="value", ax=ax)
            for x in ax.get_children():
                if isinstance(x, patches):
                    x.set_alpha(0.25)
            sns.swarmplot(data=p, x="variable", y="value", alpha=0.5, ax=ax)
            ax.set(ylim=(0, 1), xlabel="", ylabel="ROC AUC")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            fig.savefig(
                output_dir
                / f"severe-mild_prediction.{name}{label}.k{k}.{N}.svg",
                **figkws,
            )
            continue

        # Get weights
        n_vars = res[0][-1].shape[0]
        real_weights = pd.DataFrame([r[-2] for r in res])
        random_weights = pd.DataFrame([r[-1] for r in res])
        real_weights.to_csv(
            output_dir
            / f"severe-mild_prediction.{name}{label}.k{k}.{N}.weights.csv"
        )
        random_weights.to_csv(
            output_dir
            / f"severe-mild_prediction.{name}{label}.k{k}.{N}.random_weights.csv"
        )
        weights = pd.read_csv(
            output_dir
            / f"severe-mild_prediction.{name}{label}.k{k}.{N}.weights.csv",
            index_col=0,
        )
        random_weights = pd.read_csv(
            output_dir
            / f"severe-mild_prediction.{name}{label}.k{k}.{N}.random_weights.csv",
            index_col=0,
        )

        # if (weights.fillna(1) > 0).all().all():
        #     wd = np.log(real_weights.mean()) - np.log(random_weights.mean())
        # else:
        wd = real_weights.mean() - random_weights.mean()
        sign = X.join(y.rename("severity")).corr()["severity"] > 0
        wd *= sign.astype(int).replace(0, -1)

        gs_kw = dict(width_ratios=[0.2, 0.8])

        fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=gs_kw)
        ax[0].axhline(0.5, linestyle="--", color="grey")
        sns.boxenplot(data=p, x="variable", y="value", ax=ax[0])
        for x in ax[0].get_children():
            if isinstance(x, patches):
                x.set_alpha(0.25)
        sns.swarmplot(data=p, x="variable", y="value", alpha=0.5, ax=ax[0])
        ax[0].set(ylim=(0, 1), xlabel="", ylabel="ROC AUC")
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

        rank = wd.rank()
        sort = wd.sort_values().dropna()
        ax[1].scatter(rank, wd, s=5, alpha=0.5)
        for v in sort.head(10).index:
            ax[1].text(rank.loc[v], wd.loc[v], s=v, ha="left")
        for v in sort.tail(10).index:
            ax[1].text(rank.loc[v], wd.loc[v], s=v, ha="right")
        ax[1].axhline(0, linestyle="--", color="grey")
        ax[1].set(
            xlabel="Variable (rank)",
            ylabel="Feature importance\nover random(log, signed)",
        )
        fig.savefig(
            output_dir / f"severe-mild_prediction.{name}{label}.k{k}.{N}.svg",
            **figkws,
        )

        pval = (
            (scores["test_score"].median() < scores["test_score_random"]).sum()
        ) / N


# Predict all samples of a patient for patients with >=3 timepooints
clf = RandomForestClassifier()
name = str(type(clf)).split(".")[-1][:-2]

pts = (
    meta.groupby(["patient", "patient_code"])
    .size()
    .sort_values()
    .loc["Patient"]
)
pts = pts[pts >= 3].index
n = 100
fig, axes = plt.subplots(2, 4, figsize=(4 * 4, 2 * 2), sharey=True)
for ax, patient in zip(axes.flat, pts):
    # patient = "P016"
    # n = 100
    pat = meta.loc[meta["patient_code"] == patient].sort_values("time_symptoms")
    X_test = matrix.loc[
        pat.index, X.columns,
    ]

    r = Parallel(n_jobs=-1)(
        delayed(predict_patient)(
            i,
            model=RandomForestClassifier,
            X=X,
            y=y,
            X_test=X_test,
            patient=patient,
        )
        for i in range(N)
    )
    r = np.concatenate(r).reshape((n, X_test.shape[0]))

    m = np.e ** np.median(r, 0)
    l = np.e ** np.percentile(r, 5, 0)
    u = np.e ** np.percentile(r, 95, 0)

    ax.plot(pat["time_symptoms"], m, "-o")
    ax.fill_between(pat["time_symptoms"], l, u, alpha=0.2)
    ax.axhline(0.5, linestyle="--", color="grey")
    ax.set(title=patient)  # ylim=(0.35, 0.85),
fig.savefig(
    output_dir
    / f"severe-mild_prediction.{name}{label}.predict_patient_timeline.{n}.svg",
    **figkws,
)


# Try out different number of variables
N = 12

perf: Dict[int, float] = dict()
perf_q5: Dict[int, float] = dict()
perf_q95: Dict[int, float] = dict()
random_perf: Dict[int, float] = dict()
random_perf_q5: Dict[int, float] = dict()
random_perf_q95: Dict[int, float] = dict()
weights_: Dict[int, pd.Series] = dict()
random_weights_: Dict[int, pd.Series] = dict()

for k in list(range(1, 11)) + [12, 15, 20, 30, 40, 50, 75, 96]:
    print(perf)
    # Fit
    res = Parallel(n_jobs=-1)(
        delayed(fit)(i, model=RandomForestClassifier, X=X, y=y, k=k)
        for i in range(N)
    )
    # Get ROC_AUC scores only
    scores = pd.DataFrame(
        np.asarray([r[:-2] for r in res]),
        columns=[
            "train_score",
            "test_score",
            "train_score_random",
            "test_score_random",
        ],
    )
    perf[k] = scores.mean()["test_score"]
    perf_q5[k] = scores.quantile(0.05)["test_score"]
    perf_q95[k] = scores.quantile(0.95)["test_score"]
    random_perf[k] = scores.mean()["test_score_random"]
    random_perf_q5[k] = scores.quantile(0.05)["test_score_random"]
    random_perf_q95[k] = scores.quantile(0.95)["test_score_random"]

    weights_[k] = (
        pd.DataFrame([r[-2] for r in res]).sum().reindex(X.columns).fillna(0)
    )
    random_weights_[k] = (
        pd.DataFrame([r[-1] for r in res]).sum().reindex(X.columns).fillna(0)
    )


x = pd.DataFrame(
    [perf, perf_q5, perf_q95, random_perf, random_perf_q5, random_perf_q95],
    index=["mean", "q5", "q95", "random_mean", "random_q5", "random_q95"],
).T.sort_index()

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(x.index, x["mean"], "-o", alpha=0.75)
ax.fill_between(x.index, x["q5"], x["q95"], alpha=0.25, label="Real labels")
ax.plot(x.index, x["random_mean"], "-o", color="grey", alpha=0.75)
ax.fill_between(
    x.index,
    x["random_q5"],
    x["random_q95"],
    alpha=0.25,
    color="grey",
    label="Randomized labels",
)
ax.axhline(0.5, linestyle="--", color="grey")
ax.legend()
ax.set(
    xlabel="K variables selected",
    ylabel="ROC-AUC\n(mean/95 CI of 100 CV loops)",
)
ax.set_ylim((0, 1))
fig.savefig(output_dir / "prediction.performance_as_select_k.svg", **figkws)
