# =============================================================================
# 0. ENVIRONMENT - pinned libraries, versions, seeds, output folders
# =============================================================================
import os, sys, io, json, time, random, hashlib, platform, datetime, itertools, subprocess, warnings, importlib, urllib.request
import importlib.metadata as _md
warnings.filterwarnings("ignore")
T_START = time.time()

# Libraries installed by this notebook at a fixed version (versions of the reference run).
PINNED = {"scikit-learn": "1.8.0", "shap": "0.52.0"}
# Libraries used as provided by the runtime. Forcing them in Colab would require a runtime restart,
# so the reference-run versions are only recorded and compared; the numeric self-check (Section 9) is the guard.
REFERENCE_VERSIONS = {"python": "3.12.3", "numpy": "2.4.4", "pandas": "3.0.2", "scipy": "1.17.1",
                      "matplotlib": "3.10.8", "openpyxl": "3.1.5"}
SKIP_PIN = os.environ.get("SEE_SKIP_PIN", "0") == "1"            # offline test runs only
SKIP_SHAP_FIG = os.environ.get("SEE_SKIP_SHAP_FIG", "0") == "1"  # numeric core only (cross-version check)

def _ver(pkg):
    importlib.invalidate_caches()
    try: return _md.version(pkg)
    except _md.PackageNotFoundError: return None

def _pip_install(spec):
    base = [sys.executable, "-m", "pip", "install", "-q", spec]
    for extra in ([], ["--break-system-packages"]):
        if subprocess.run(base + extra, capture_output=True, text=True).returncode == 0:
            return True
    return False

PIN_LOG = []
for _pkg, _want in PINNED.items():
    if SKIP_SHAP_FIG and _pkg == "shap":
        continue
    _have = _ver(_pkg)
    if _have == _want:
        PIN_LOG.append(f"{_pkg}=={_want}: already installed"); continue
    if SKIP_PIN:
        PIN_LOG.append(f"{_pkg}: pin skipped, found {_have}"); continue
    if _pkg == "scikit-learn" and "sklearn" in sys.modules:
        PIN_LOG.append("WARNING: sklearn was imported before pinning. Restart the runtime and run all cells again.")
    _ok = _pip_install(f"{_pkg}=={_want}")
    if not _ok and _have is None:
        _ok = _pip_install(_pkg)
    PIN_LOG.append(f"{_pkg}: wanted {_want}, found {_have}, now {_ver(_pkg)}" + ("" if _ok else "  (pip failed)"))
if _ver("openpyxl") is None:
    _pip_install("openpyxl")
importlib.invalidate_caches()

import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

VERSIONS = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "matplotlib": _ver("matplotlib"), "shap": _ver("shap"), "openpyxl": _ver("openpyxl")}
VERSION_NOTES = []
for _k, _v in {**REFERENCE_VERSIONS, **PINNED}.items():
    if VERSIONS.get(_k) != _v:
        VERSION_NOTES.append(f"{_k}: this run {VERSIONS.get(_k)}, reference run {_v}")

SEED0 = 42                     # primary data partition and model seed
SEEDS = list(range(42, 52))    # ten data partitions for the sensitivity analysis
N_SPLITS = 10
random.seed(SEED0); np.random.seed(SEED0)

RES, CSV, FIG, DATA = "results", os.path.join("results", "csv"), "figures", "data"
for _d in (RES, CSV, FIG, DATA):
    os.makedirs(_d, exist_ok=True)

for _l in PIN_LOG: print("[pin]", _l)
print("[versions]", ", ".join(f"{k} {v}" for k, v in VERSIONS.items()))
print("[versions] identical to the reference run" if not VERSION_NOTES else "[versions] differences from the reference run:\n   " + "\n   ".join(VERSION_NOTES))

# =============================================================================
# II-A. DATASETS - download, integrity check, identical preprocessing, Table I
# =============================================================================
ORDINAL = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}
DRIVERS = ["rely", "data", "cplx", "time", "stor", "virt", "turn", "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced"]
_OURMINE = "https://raw.githubusercontent.com/timm/ourmine/master/our/arffs/effest/"
DATASETS = {
    "COCOMO81": dict(file="coc81_1_1.arff", url=_OURMINE + "coc81_1_1.arff", md5="2898f9dca7709ba5dc6384500e97a5f6",
                     target="actual", unit="person-months", drop=["project_id", "dev_mode"], onehot=[], ordinal=[],
                     size=["loc"], columns=DRIVERS + ["loc"]),
    "NASA93": dict(file="nasa93.arff", url=_OURMINE + "nasa93.arff", md5="6bfcb7491cb8c592a7e719a50e086167",
                   target="act_effort", unit="person-months",
                   drop=["recordnumber", "projectname", "cat2", "forg", "center", "year"], onehot=["mode"], ordinal=DRIVERS,
                   size=["equivphyskloc"], columns=DRIVERS + ["equivphyskloc", "mode_organic", "mode_semidetached"]),
    "Desharnais": dict(file="desharnais.arff.txt",
                       url="https://raw.githubusercontent.com/deepak21188/Software-Development-Effort-Estimation-using-ML/master/desharnais.arff.txt",
                       md5="59c5b1a1978986239793ea4c78c504fd", target="Effort", unit="person-hours",
                       drop=["Project"], onehot=["Language"], ordinal=[], size=["PointsAdjust", "PointsNonAjust"],
                       columns=["TeamExp", "ManagerExp", "YearEnd", "Length", "Transactions", "Entities",
                                "PointsAdjust", "Envergure", "PointsNonAjust", "Language_2", "Language_3"]),
}
DS_LIST = list(DATASETS)

def _resolve(name, cols):
    return {c.lower(): c for c in cols}.get(name.lower())

def fetch(cfg):
    """Use data/<file> if present; otherwise download it once. Returns raw bytes and md5."""
    path = os.path.join(DATA, cfg["file"])
    if not os.path.exists(path):
        req = urllib.request.Request(cfg["url"], headers={"User-Agent": "Mozilla/5.0"})
        with open(path, "wb") as fh:
            fh.write(urllib.request.urlopen(req, timeout=60).read())
    raw = open(path, "rb").read()
    return raw, hashlib.md5(raw).hexdigest()

def load_dataset(name):
    cfg = DATASETS[name]
    raw, digest = fetch(cfg)
    rec, _ = arff.loadarff(io.StringIO(raw.decode("utf-8", "replace")))
    df = pd.DataFrame(rec)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:                                   # nominal attributes arrive as bytes
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v).replace("?", np.nan)
    for col in cfg["ordinal"]:                             # ordinal cost drivers -> 1..6
        cc = _resolve(col, df.columns)
        mapped = df[cc].astype(str).str.strip().str.lower().map(ORDINAL)
        assert not mapped.isna().any(), f"{name}: unmapped ordinal level in {cc}"
        df[cc] = mapped
    n_raw = len(df)
    df = df.dropna().reset_index(drop=True)                # Desharnais: 4 projects with missing values
    target = _resolve(cfg["target"], df.columns)
    y = df[target].astype(float).values
    dropped = [_resolve(c, df.columns) for c in cfg["drop"]]
    X = df.drop(columns=[target] + dropped)
    if cfg["onehot"]:
        X = pd.get_dummies(X, columns=[_resolve(c, X.columns) for c in cfg["onehot"]], drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    assert not X.isna().any().any(), f"{name}: non-numeric feature values"
    assert list(X.columns) == cfg["columns"], f"{name}: unexpected column order {list(X.columns)}"   # tree models depend on column order
    X = X.astype(float)
    info = dict(dataset=name, projects_raw=n_raw, projects_used=len(df), features=X.shape[1], effort_unit=cfg["unit"],
                size_features=", ".join(cfg["size"]), dropped_columns=", ".join(dropped),
                md5=digest, md5_matches_reference=(digest == cfg["md5"]), source_url=cfg["url"])
    return X, y, info

from decimal import Decimal, getcontext
getcontext().prec = 50
def log1p_exact(values):
    """Correctly rounded ln(1 + v). numpy's log1p differs by one unit in the last place between CPUs (vectorized SVML
    kernel on AVX-512 machines, C library elsewhere); in tree induction that is enough to flip near-tied splits, so the
    transformation is computed here in a platform-independent way."""
    flat = [float((Decimal(1) + Decimal(float(v))).ln()) for v in np.ravel(values)]
    return np.array(flat, dtype=float).reshape(np.shape(values))

DATA_X, DATA_Y, Y_LOG, LOG1P_PLATFORM_DIFF, _info, _desc = {}, {}, {}, {}, [], []
for ds in DS_LIST:
    X, y, info = load_dataset(ds)
    DATA_X[ds], DATA_Y[ds], Y_LOG[ds] = X, y, log1p_exact(y)
    LOG1P_PLATFORM_DIFF[ds] = int((np.log1p(y) != Y_LOG[ds]).sum())      # > 0 on platforms whose log1p is not correctly rounded
    _info.append(info)
    row = dict(dataset=ds, n=len(y), effort_min=y.min(), effort_median=float(np.median(y)), effort_mean=y.mean(), effort_max=y.max(),
               skewness=float(stats.skew(y)), skewness_log1p=float(stats.skew(Y_LOG[ds])))
    for s in DATASETS[ds]["size"]:
        row[f"size_min[{s}]"], row[f"size_max[{s}]"] = float(X[s].min()), float(X[s].max())
    _desc.append(row)
T1_DATASETS = pd.DataFrame(_info)
DESCRIPTIVES = pd.DataFrame(_desc)
print(T1_DATASETS[["dataset", "projects_raw", "projects_used", "features", "effort_unit", "size_features", "md5_matches_reference"]].to_string(index=False))
print("targets where this platform's np.log1p deviates from the correctly rounded value:", LOG1P_PLATFORM_DIFF)
print(DESCRIPTIVES[["dataset", "n", "effort_median", "effort_mean", "effort_max", "skewness", "skewness_log1p"]].round(2).to_string(index=False))

# =============================================================================
# II-B. EVALUATION METRICS (computed on the original effort scale, per test fold)
# =============================================================================
METRICS = ["MAE", "RMSE", "MMRE", "MdMRE", "PRED25", "SA"]
LOWER_IS_BETTER = {"MAE": True, "RMSE": True, "MMRE": True, "MdMRE": True, "PRED25": False, "SA": False}
DECIMALS = {"MAE": 1, "RMSE": 1, "MMRE": 2, "MdMRE": 2, "PRED25": 1, "SA": 1}   # precision displayed in the manuscript

PRED_TOL = 1e-9   # a relative error of exactly 0.25 must count as a hit on every platform (the exp/log round trip is exact only up to the last bit)

def fold_metrics(y_true, y_pred, y_train):
    ae = np.abs(y_true - y_pred)
    mre = ae / np.clip(np.abs(y_true), 1e-9, None)
    # Exact MAE of random guessing P0: mean |y_i - y_j| over all test-train project pairs
    mae_p0 = np.mean([np.mean(np.abs(yi - y_train)) for yi in y_true])
    return dict(MAE=float(ae.mean()), RMSE=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                MMRE=float(mre.mean()), MdMRE=float(np.median(mre)),
                PRED25=float(100.0 * np.mean(mre <= 0.25 + PRED_TOL)), SA=float(100.0 * (1.0 - ae.mean() / mae_p0)))

# =============================================================================
# II-D. EXPERIMENTAL PROTOCOL - five algorithms, three model specifications,
#       paired 10-fold cross-validation, ten data partitions (seeds 42-51)
# =============================================================================
ALGOS = ["LR", "DT", "RF", "KNN", "SVR"]
ALGO_NAMES = {"LR": "Linear Regression", "DT": "Decision Tree", "RF": "Random Forest", "KNN": "K-Nearest Neighbors", "SVR": "Support Vector Regression"}
# raw     : no transformation
# semilog : ln(1+y) target only            (primary protocol; identical to the previous submission)
# loglog  : ln(1+y) target and ln(1+size)  (size = KLOC or function points)
SPECS = {"raw": dict(log_target=False, log_size=False),
         "semilog": dict(log_target=True, log_size=False),
         "loglog": dict(log_target=True, log_size=True)}
PRIMARY = "semilog"

def make_models():
    return {"LR": LinearRegression(), "DT": DecisionTreeRegressor(random_state=SEED0),
            "RF": RandomForestRegressor(n_estimators=100, random_state=SEED0),
            "KNN": KNeighborsRegressor(n_neighbors=5), "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1)}

def make_estimator(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])

def design_matrix(Xdf, size_cols, log_size):
    X = Xdf.values.astype(float).copy()
    if log_size:                                      # fixed element-wise function: no information leaks across folds
        idx = [list(Xdf.columns).index(c) for c in size_cols]
        X[:, idx] = log1p_exact(X[:, idx])
    return X

def run_cv(X, y, y_log, seed, log_target):
    """The target is ln(1+y) during training and predictions are mapped back with exp(.)-1 before any metric is computed."""
    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X))   # shared by all algorithms -> paired folds
    per_fold = {a: {m: [] for m in METRICS} for a in ALGOS}
    oof = {a: np.zeros(len(y)) for a in ALGOS}
    fold_id = np.zeros(len(y), dtype=int)
    for a, model in make_models().items():
        for k, (tr, te) in enumerate(folds):
            est = make_estimator(model)
            est.fit(X[tr], y_log[tr] if log_target else y[tr])
            pred = np.expm1(est.predict(X[te])) if log_target else est.predict(X[te])
            oof[a][te] = pred; fold_id[te] = k
            for m, v in fold_metrics(y[te], pred, y[tr]).items():
                per_fold[a][m].append(v)
    return per_fold, oof, fold_id

FOLD = {}   # FOLD[(dataset, spec, seed)][algo][metric] -> list of 10 per-fold values
LR_WORST = []
OOF = {}    # OOF[(dataset, spec)] -> out-of-fold predictions for the primary partition (seed 42)
_t = time.time()
for ds in DS_LIST:
    for spec, opt in SPECS.items():
        X = design_matrix(DATA_X[ds], DATASETS[ds]["size"], opt["log_size"])
        for seed in SEEDS:
            pf, oof, fid = run_cv(X, DATA_Y[ds], Y_LOG[ds], seed, opt["log_target"])
            FOLD[(ds, spec, seed)] = pf
            _ae = np.abs(DATA_Y[ds] - oof["LR"]); _i = int(np.argmax(_ae))       # project with the largest LR error in this partition
            LR_WORST.append(dict(dataset=ds, spec=spec, seed=seed, LR_mean_MAE=float(np.mean(pf["LR"]["MAE"])),
                                 LR_max_fold_MAE=float(np.max(pf["LR"]["MAE"])), worst_project_row=_i,
                                 worst_project_actual=float(DATA_Y[ds][_i]), worst_project_pred=float(oof["LR"][_i]),
                                 **{f"size[{c}]": float(DATA_X[ds][c].iloc[_i]) for c in DATASETS[ds]["size"]}))
            if seed == SEED0:
                OOF[(ds, spec)] = dict(pred=oof, fold=fid)
    print(f"{ds}: 3 specifications x {len(SEEDS)} partitions x {N_SPLITS} folds done ({time.time() - _t:.0f} s elapsed)")

# Long-format records
PER_FOLD = pd.DataFrame([dict(dataset=ds, spec=spec, algorithm=a, fold=k + 1, **{m: FOLD[(ds, spec, SEED0)][a][m][k] for m in METRICS})
                         for ds in DS_LIST for spec in SPECS for a in ALGOS for k in range(N_SPLITS)])
PER_SEED = pd.DataFrame([dict(dataset=ds, spec=spec, algorithm=a, seed=seed, **{m: float(np.mean(FOLD[(ds, spec, seed)][a][m])) for m in METRICS})
                         for ds in DS_LIST for spec in SPECS for a in ALGOS for seed in SEEDS])
print("per_fold rows:", len(PER_FOLD), "| per_seed rows:", len(PER_SEED))

# =============================================================================
# III-A/B/C. RESULT TABLES - main performance, specification ablation, stability, LR failure anatomy
# =============================================================================
def _best(values, metric):
    r = np.round(np.asarray(values, float), DECIMALS[metric])
    return r == (r.min() if LOWER_IS_BETTER[metric] else r.max())     # ties at the displayed precision are all flagged

# ---- Table II: primary protocol (semi-log, partition seed 42); mean and sample SD (ddof = 1) across the 10 folds
rows = []
for ds in DS_LIST:
    block = {a: FOLD[(ds, PRIMARY, SEED0)][a] for a in ALGOS}
    flags = {m: _best([np.mean(block[a][m]) for a in ALGOS], m) for m in METRICS}
    for i, a in enumerate(ALGOS):
        row = dict(dataset=ds, algorithm=a)
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_sd"] = float(np.mean(block[a][m])), float(np.std(block[a][m], ddof=1))
            row[f"{m}_median_fold"] = float(np.median(block[a][m]))
            row[f"{m}_best"] = bool(flags[m][i])
        rows.append(row)
T2_MAIN = pd.DataFrame(rows)

# ---- Ablation over specifications and partitions (all metrics, long format)
rows = []
for (ds, a, spec), g in PER_SEED.groupby(["dataset", "algorithm", "spec"], sort=False):
    for m in METRICS:
        v = g[m].values
        rows.append(dict(dataset=ds, algorithm=a, spec=spec, metric=m, seed42=float(g.loc[g.seed == SEED0, m].iloc[0]),
                         mean10=float(v.mean()), sd10=float(v.std(ddof=1)), min10=float(v.min()), max10=float(v.max())))
ABLATION = pd.DataFrame(rows)

# ---- Table III: MAE by specification, mean and sample SD over the 10 partitions (wide)
_w = ABLATION[ABLATION.metric == "MAE"].pivot_table(index=["dataset", "algorithm"], columns="spec", values=["seed42", "mean10", "sd10"], sort=False)
_w.columns = [f"{spec}_{stat}" for stat, spec in _w.columns]
T3_ABLATION = _w.reset_index()
T3_ABLATION["dataset"] = pd.Categorical(T3_ABLATION["dataset"], DS_LIST); T3_ABLATION["algorithm"] = pd.Categorical(T3_ABLATION["algorithm"], ALGOS)
T3_ABLATION = T3_ABLATION.sort_values(["dataset", "algorithm"]).reset_index(drop=True)
T3_ABLATION = T3_ABLATION[["dataset", "algorithm"] + [f"{s}_{k}" for s in SPECS for k in ("seed42", "mean10", "sd10")]]
for s in ("raw", "loglog"):
    T3_ABLATION[f"ratio_semilog_over_{s}_mean10"] = T3_ABLATION["semilog_mean10"] / T3_ABLATION[f"{s}_mean10"]

# ---- Table IV: stability of the primary protocol across the 10 partitions
rows = []
for ds in DS_LIST:
    for a in ALGOS:
        v = PER_SEED[(PER_SEED.dataset == ds) & (PER_SEED.spec == PRIMARY) & (PER_SEED.algorithm == a)].sort_values("seed")["MAE"].values
        rows.append(dict(dataset=ds, algorithm=a, MAE_mean10=float(v.mean()), MAE_sd10=float(v.std(ddof=1)), MAE_cv_percent=float(100 * v.std(ddof=1) / v.mean()),
                         MAE_min10=float(v.min()), MAE_max10=float(v.max()), MAE_seed42=float(v[0]), seed42_position_low_to_high=int((v < v[0]).sum()) + 1))
T4_STABILITY = pd.DataFrame(rows)

# ---- Rank summary (supports a rank-based definition of consistency)
rows = []
for ds in DS_LIST:
    for spec in SPECS:
        mat = np.array([[PER_SEED[(PER_SEED.dataset == ds) & (PER_SEED.spec == spec) & (PER_SEED.algorithm == a) & (PER_SEED.seed == sd)]["MAE"].iloc[0] for a in ALGOS] for sd in SEEDS])
        rk = np.vstack([stats.rankdata(r) for r in mat])
        for j, a in enumerate(ALGOS):
            rows.append(dict(dataset=ds, spec=spec, algorithm=a, mean_rank_MAE_over_partitions=float(rk[:, j].mean()), times_ranked_first=int((rk[:, j] == 1).sum())))
RANK_SUMMARY = pd.DataFrame(rows)
_cells = []
for ds in DS_LIST:                                   # 3 datasets x 6 metrics at the primary partition
    for m in METRICS:
        v = np.array([T2_MAIN[(T2_MAIN.dataset == ds) & (T2_MAIN.algorithm == a)][f"{m}_mean"].iloc[0] for a in ALGOS])
        _cells.append(stats.rankdata(np.round(v if LOWER_IS_BETTER[m] else -v, 10)))
_cells = np.vstack(_cells)
RANK_18 = pd.DataFrame(dict(algorithm=ALGOS, mean_rank_18_cells=_cells.mean(axis=0), cells_ranked_first=(_cells <= 1.5).sum(axis=0)))

# ---- Anatomy of the Linear Regression failure
LR_WORST_DF = pd.DataFrame(LR_WORST)
rows = []
for ds in DS_LIST:
    lr = np.array(FOLD[(ds, PRIMARY, SEED0)]["LR"]["MAE"]); k = int(np.argmax(lr))
    idx = np.where(OOF[(ds, PRIMARY)]["fold"] == k)[0]
    for i in idx:
        rows.append(dict(dataset=ds, worst_fold=k + 1, fold_MAE_LR_semilog=float(lr[k]), median_fold_MAE_LR_semilog=float(np.median(lr)), project_row=int(i),
                         **{f"size[{c}]": float(DATA_X[ds][c].iloc[i]) for c in DATASETS[ds]["size"]}, actual=float(DATA_Y[ds][i]),
                         **{f"LR_pred_{s}": float(OOF[(ds, s)]["pred"]["LR"][i]) for s in SPECS}, RF_pred_semilog=float(OOF[(ds, PRIMARY)]["pred"]["RF"][i])))
LR_CASE = pd.DataFrame(rows)

pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print("Table II (means):"); print(T2_MAIN[["dataset", "algorithm"] + [f"{m}_mean" for m in METRICS]].to_string(index=False))
print("\nTable III (MAE, mean over 10 partitions):"); print(T3_ABLATION[["dataset", "algorithm", "raw_mean10", "semilog_mean10", "loglog_mean10"]].to_string(index=False))

# =============================================================================
# III-D. STATISTICAL INFERENCE - Friedman, Nemenyi (critical difference), Wilcoxon-Holm per fold and per project, effect sizes
# =============================================================================
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}     # Demsar (2006), Nemenyi, alpha = 0.05
K_ALG = len(ALGOS)
CD = Q_ALPHA_05[K_ALG] * np.sqrt(K_ALG * (K_ALG + 1) / (6.0 * N_SPLITS))
PAIRS = list(itertools.combinations(ALGOS, 2))

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); running = 0.0
    for rank_i, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank_i) * p[idx])); adj[idx] = running
    return adj

def _wilcoxon(a, b, **kw):
    try: return float(stats.wilcoxon(a, b, **kw).pvalue)
    except ValueError: return 1.0                                   # all paired differences are zero

def a12_lower(x, y):
    """Vargha-Delaney A12 for errors: probability that an error drawn from x is lower than one drawn from y."""
    x = np.asarray(x, float)[:, None]; y = np.asarray(y, float)[None, :]
    return float((x < y).mean() + 0.5 * (x == y).mean())

def _magnitude(a):
    d = max(a, 1 - a)
    return "large" if d >= 0.71 else "medium" if d >= 0.64 else "small" if d >= 0.56 else "negligible"

fr_rows, nem_rows, pw_rows = [], [], []
for spec in (PRIMARY, "loglog"):
    for ds in DS_LIST:
        mae = np.array([FOLD[(ds, spec, SEED0)][a]["MAE"] for a in ALGOS]).T          # folds x algorithms
        fr = stats.friedmanchisquare(*[mae[:, j] for j in range(K_ALG)])
        avg_rank = np.vstack([stats.rankdata(r) for r in mae]).mean(axis=0)
        fr_rows.append(dict(spec=spec, dataset=ds, friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue), CD=float(CD),
                            **{f"avg_rank_{a}": float(avg_rank[j]) for j, a in enumerate(ALGOS)}))
        se = np.sqrt(K_ALG * (K_ALG + 1) / (6.0 * N_SPLITS))
        ae = {a: np.abs(DATA_Y[ds] - OOF[(ds, spec)]["pred"][a]) for a in ALGOS}      # out-of-fold absolute error per project
        p_fold = [_wilcoxon(mae[:, ALGOS.index(a)], mae[:, ALGOS.index(b)]) for a, b in PAIRS]
        p_proj = [_wilcoxon(ae[a], ae[b], method="approx") for a, b in PAIRS]
        h_fold, h_proj = holm(p_fold), holm(p_proj)
        for n, (a, b) in enumerate(PAIRS):
            diff = abs(avg_rank[ALGOS.index(a)] - avg_rank[ALGOS.index(b)])
            nem_rows.append(dict(spec=spec, dataset=ds, pair=f"{a}-{b}", rank_difference=float(diff), exceeds_CD=bool(diff > CD),
                                 nemenyi_p=float(stats.studentized_range.sf(diff / se * np.sqrt(2.0), K_ALG, np.inf))))
            A = a12_lower(ae[a], ae[b])
            pw_rows.append(dict(spec=spec, dataset=ds, pair=f"{a}-{b}", n_folds=N_SPLITS, p_fold_raw=p_fold[n], p_fold_holm=float(h_fold[n]),
                                n_projects=len(DATA_Y[ds]), p_project_raw=p_proj[n], p_project_holm=float(h_proj[n]),
                                median_AE_first=float(np.median(ae[a])), median_AE_second=float(np.median(ae[b])),
                                A12_first_lower_error=A, cliffs_delta=2 * A - 1, magnitude=_magnitude(A),
                                paired_win_rate_first=float(np.mean(ae[a] < ae[b]) + 0.5 * np.mean(ae[a] == ae[b]))))
FRIEDMAN, NEMENYI, WILCOXON_HOLM = pd.DataFrame(fr_rows), pd.DataFrame(nem_rows), pd.DataFrame(pw_rows)
EFFECT_SIZES = WILCOXON_HOLM[["spec", "dataset", "pair", "median_AE_first", "median_AE_second", "A12_first_lower_error", "cliffs_delta", "magnitude", "paired_win_rate_first"]].copy()

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(f"CD = {CD:.3f}"); print(FRIEDMAN.to_string(index=False))
for spec in (PRIMARY, "loglog"):
    w = WILCOXON_HOLM[WILCOXON_HOLM.spec == spec]
    print(f"\n[{spec}] pairs significant after Holm (alpha = 0.05): per fold {int((w.p_fold_holm < 0.05).sum())}/30 | per project {int((w.p_project_holm < 0.05).sum())}/30")
    print(w[w.p_project_holm < 0.05][["dataset", "pair", "p_fold_holm", "p_project_holm", "A12_first_lower_error", "magnitude"]].to_string(index=False))

# =============================================================================
# III-E. SHAP INTERPRETATION - Random Forest on ln(1+effort), TreeExplainer
# =============================================================================
SHAP_VALUES = {}
rows = []
if not SKIP_SHAP_FIG:
    import shap
    for ds in DS_LIST:
        Xdf, y = DATA_X[ds], DATA_Y[ds]
        Xs = StandardScaler().fit_transform(Xdf.values)
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED0).fit(Xs, Y_LOG[ds])
        try:
            sv = shap.TreeExplainer(rf).shap_values(Xs)
        except Exception:
            sv = shap.TreeExplainer(rf).shap_values(Xs, check_additivity=False)
        sv = np.asarray(sv, float)
        SHAP_VALUES[ds] = sv
        imp = np.abs(sv).mean(axis=0); order = np.argsort(-imp)
        for r, j in enumerate(order, start=1):
            rho = stats.spearmanr(Xdf.values[:, j], sv[:, j])[0] if np.ptp(sv[:, j]) > 0 else np.nan
            rows.append(dict(dataset=ds, rank=r, feature=Xdf.columns[j], mean_abs_shap=float(imp[j]), share_percent=float(100 * imp[j] / imp.sum()),
                             spearman_feature_vs_shap=float(rho)))
SHAP_IMPORTANCE = pd.DataFrame(rows, columns=["dataset", "rank", "feature", "mean_abs_shap", "share_percent", "spearman_feature_vs_shap"])
if len(SHAP_IMPORTANCE):
    print(SHAP_IMPORTANCE[SHAP_IMPORTANCE["rank"] <= 5].to_string(index=False))

# =============================================================================
# FIGURES - vector PDF + 600-dpi PNG; serif font bundled with matplotlib (identical on every platform);
#           one color, marker and hatch per algorithm in every figure; no in-figure titles, panel labels only
# =============================================================================
FIGURE_FILES = []
if not SKIP_SHAP_FIG:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9.5, "legend.fontsize": 10, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                         "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    W2, W1 = 7.0, 3.4                                   # text width and column width of the A4 two-column template (inches)
    COLORS = {"LR": "#0072B2", "DT": "#E69F00", "RF": "#009E73", "KNN": "#CC79A7", "SVR": "#D55E00"}   # Okabe-Ito, color-blind safe
    MARKERS = {"LR": "o", "DT": "s", "RF": "^", "KNN": "D", "SVR": "v"}
    HATCHES = {"LR": "", "DT": "////", "RF": "\\\\\\\\", "KNN": "xxxx", "SVR": "...."}
    UNIT = {ds: DATASETS[ds]["unit"] for ds in DS_LIST}
    PANEL = dict(zip(DS_LIST, ["(a) COCOMO81", "(b) NASA93", "(c) Desharnais"]))

    def _save(fig, name):
        for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
            fig.savefig(os.path.join(FIG, f"{name}.{ext}"), **kw)
        plt.close(fig); FIGURE_FILES.append(name)

    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
    def _log_y(ax):
        """Label half-decade (or 2-3-5) minor ticks when the axis spans few decades, so that every panel shows at least three labels."""
        ax.set_yscale("log"); lo, hi = ax.get_ylim(); span = np.log10(hi / lo)
        subs = (2.0, 3.0, 5.0) if span < 1.5 else (3.0,) if span < 2.6 else None
        if subs is None:
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs))
            ax.yaxis.set_minor_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(10, 10)))

    def _panel_label(ax, text):
        ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)

    # ---- Fig. 3: distribution of per-fold MAE under the primary protocol
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.0))
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, DS_LIST):
        data = [FOLD[(ds, PRIMARY, SEED0)][a]["MAE"] for a in ALGOS]
        bp = ax.boxplot(data, widths=0.58, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.1),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        for patch, a in zip(bp["boxes"], ALGOS):
            patch.set_facecolor("white"); patch.set_edgecolor(COLORS[a]); patch.set_hatch(HATCHES[a])
        for i, a in enumerate(ALGOS):
            ax.scatter(rng.normal(i + 1, 0.07, len(data[i])), data[i], s=22, marker=MARKERS[a], facecolor=COLORS[a], edgecolor="black", linewidth=0.3, zorder=3)
        _log_y(ax); ax.set_xticks(range(1, 6)); ax.set_xticklabels(ALGOS); ax.grid(axis="y", which="major", ls=":", lw=0.4, alpha=0.7)
        ax.set_ylabel(f"MAE per fold ({UNIT[ds]})"); _panel_label(ax, PANEL[ds])
    fig.tight_layout(w_pad=1.0); _save(fig, "Fig3_perfold_mae")

    # ---- Fig. 4: specification ablation, mean MAE over ten partitions with min-max range
    SPEC_STYLE = {"raw": dict(marker="o", mfc="white", mec="black", label="No transformation"),
                  "semilog": dict(marker="s", mfc="#555555", mec="black", label="Semi-log (target only)"),
                  "loglog": dict(marker="^", mfc="#0072B2", mec="black", label="Log-log (target and size)")}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.25))
    for ax, ds in zip(axes, DS_LIST):
        for off, spec in zip((-0.24, 0.0, 0.24), SPECS):
            sub = ABLATION[(ABLATION.dataset == ds) & (ABLATION.spec == spec) & (ABLATION.metric == "MAE")].set_index("algorithm").loc[ALGOS]
            x = np.arange(1, 6) + off; st = SPEC_STYLE[spec]
            ax.vlines(x, sub["min10"], sub["max10"], color="#777777", lw=1.0, zorder=2)
            ax.plot(x, sub["mean10"], ls="none", marker=st["marker"], ms=6.5, mfc=st["mfc"], mec=st["mec"], mew=0.5, zorder=3)
        _log_y(ax); ax.set_xticks(range(1, 6)); ax.set_xticklabels(ALGOS); ax.grid(axis="y", which="major", ls=":", lw=0.4, alpha=0.7)
        ax.set_ylabel(f"MAE ({UNIT[ds]})"); _panel_label(ax, PANEL[ds])
    handles = [Line2D([], [], ls="none", marker=s["marker"], ms=6.5, mfc=s["mfc"], mec=s["mec"], mew=0.5, label=s["label"]) for s in SPEC_STYLE.values()]
    handles.append(Line2D([], [], color="#777777", lw=1.0, label="Min-max over 10 partitions"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.07, 1, 1)); _save(fig, "Fig4_specification_ablation")

    # ---- Fig. 5: critical-difference diagrams (average ranks of per-fold MAE; bars join algorithms that do not differ at alpha = 0.05)
    def _cd_panel(ax, ranks, label):
        order = sorted(ALGOS, key=lambda a: ranks[a]); y_ax = 0.64
        ax.set_xlim(0.75, 5.25); ax.set_ylim(-0.06, 1.08); ax.axis("off")
        ax.plot([1, 5], [y_ax, y_ax], color="black", lw=0.7)
        for t in range(1, 6):
            ax.plot([t, t], [y_ax, y_ax + 0.035], color="black", lw=0.7); ax.text(t, y_ax + 0.05, str(t), ha="center", va="bottom", fontsize=10)
        ax.plot([1, 1 + CD], [1.0, 1.0], color="black", lw=1.2)
        for xe in (1, 1 + CD): ax.plot([xe, xe], [0.975, 1.025], color="black", lw=0.7)
        ax.text(1 + CD + 0.08, 1.0, f"CD = {CD:.2f}", ha="left", va="center", fontsize=10)
        groups = []                                                   # maximal runs of adjacent algorithms within one CD
        for i in range(len(order)):
            j = i
            while j + 1 < len(order) and ranks[order[j + 1]] - ranks[order[i]] < CD: j += 1
            if j > i and not any(i >= g[0] and j <= g[1] for g in groups): groups.append((i, j))
        for n, (i, j) in enumerate(groups):
            yb = y_ax - 0.07 - 0.075 * n
            ax.plot([ranks[order[i]] - 0.04, ranks[order[j]] + 0.04], [yb, yb], color="black", lw=2.6, solid_capstyle="butt")
        left, right = order[:3], order[3:][::-1]
        for side, names in (("left", left), ("right", right)):
            for n, a in enumerate(names):
                yl = 0.34 - 0.17 * n; xr = ranks[a]; xl = 0.95 if side == "left" else 5.05
                ax.plot([xr, xr, xl], [y_ax, yl, yl], color=COLORS[a], lw=1.1)
                ax.plot([xr], [y_ax], marker=MARKERS[a], ms=6.0, mfc=COLORS[a], mec="black", mew=0.4, zorder=4)
                ax.text(xl - 0.04 if side == "left" else xl + 0.04, yl, f"{a} ({ranks[a]:.2f})", ha="right" if side == "left" else "left", va="center", fontsize=10)
        ax.text(0.0, 1.16, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)

    def _cd_figure(specs, width, name):
        fig, axes = plt.subplots(3, len(specs), figsize=(width, 4.9), squeeze=False)
        for c, spec in enumerate(specs):
            for r, ds in enumerate(DS_LIST):
                f = FRIEDMAN[(FRIEDMAN.spec == spec) & (FRIEDMAN.dataset == ds)].iloc[0]
                tag = PANEL[ds] if len(specs) == 1 else f"({'abcdef'[c * 3 + r]}) {ds}, {'semi-log' if spec == 'semilog' else 'log-log'}"
                ptxt = "$p$ < 0.001" if f["friedman_p"] < 0.001 else f"$p$ = {f['friedman_p']:.3f}"
                _cd_panel(axes[r, c], {a: f[f"avg_rank_{a}"] for a in ALGOS}, f"{tag}: Friedman {ptxt}")
        fig.tight_layout(h_pad=2.2, w_pad=2.0); _save(fig, name)
    _cd_figure([PRIMARY, "loglog"], W2, "Fig5_cd_diagrams")
    _cd_figure([PRIMARY], W1, "Fig5alt_cd_diagrams_primary_only")

    # ---- Fig. 6: SHAP summary (top-8 features per dataset); color = feature value, cividis stays monotonic in grayscale
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.4))
    cmap = plt.get_cmap("cividis"); rng = np.random.default_rng(0)
    for ax, ds in zip(axes, DS_LIST):
        sv, Xdf = SHAP_VALUES[ds], DATA_X[ds]
        top = list(SHAP_IMPORTANCE[(SHAP_IMPORTANCE.dataset == ds) & (SHAP_IMPORTANCE["rank"] <= 8)]["feature"])
        for row, feat in enumerate(top):
            j = list(Xdf.columns).index(feat); v = sv[:, j]; fv = Xdf.values[:, j]
            col = stats.rankdata(fv) / len(fv) if np.ptp(fv) > 0 else np.full(len(fv), 0.5)
            bins = np.digitize(v, np.linspace(v.min(), v.max() + 1e-12, 28)); off = np.zeros(len(v))
            for b in np.unique(bins):
                idx = np.where(bins == b)[0]; k = np.arange(len(idx)); off[idx] = ((k + 1) // 2) * np.where(k % 2 == 0, 1, -1) * 0.055
            ypos = len(top) - 1 - row + np.clip(off, -0.36, 0.36)
            ax.scatter(v, ypos, c=col, cmap=cmap, vmin=0, vmax=1, s=13, linewidths=0.2, edgecolors="#333333", zorder=3)
        ax.axvline(0, color="#888888", lw=0.6, zorder=1)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top[::-1]); ax.set_ylim(-0.6, len(top) - 0.4)
        ax.set_xlabel("SHAP value, ln(1 + effort)"); _panel_label(ax, PANEL[ds]); ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.24, 0.012, 0.58]); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value", labelpad=-8); cb.outline.set_linewidth(0.4)
    _save(fig, "Fig6_shap_summary")

    # ---- Optional: empirical view of the extrapolation failure (COCOMO81, fold containing the largest project)
    ds = "COCOMO81"; case = LR_CASE[LR_CASE.dataset == ds]; k = int(case["worst_fold"].iloc[0]) - 1
    te = OOF[(ds, PRIMARY)]["fold"] == k; size = DATA_X[ds]["loc"].values; y = DATA_Y[ds]
    fig, ax = plt.subplots(figsize=(W1, 3.3))
    ax.scatter(size[~te], y[~te], s=16, marker="o", facecolor="#BBBBBB", edgecolor="none", label="Training projects (actual)", zorder=2)
    ax.scatter(size[te], y[te], s=16, marker="o", facecolor="black", edgecolor="black", label="Test projects (actual)", zorder=4)
    ax.scatter(size[te], np.clip(OOF[(ds, "semilog")]["pred"]["LR"][te], 1e-1, None), s=18, marker="s", facecolor="#555555", edgecolor="black", linewidth=0.4, label="LR prediction, semi-log", zorder=3)
    ax.scatter(size[te], np.clip(OOF[(ds, "loglog")]["pred"]["LR"][te], 1e-1, None), s=20, marker="^", facecolor="#0072B2", edgecolor="black", linewidth=0.4, label="LR prediction, log-log", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Size (KLOC)"); ax.set_ylabel("Effort (person-months)")
    ax.grid(ls=":", lw=0.4, alpha=0.7); ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(); _save(fig, "FigOpt_extrapolation_case_cocomo81")
    print("figures written:", ", ".join(FIGURE_FILES))

# =============================================================================
# 9. EXPORT - Excel workbook, CSV files, number trace, run log, self-check against the reference run
# =============================================================================
w = WILCOXON_HOLM
INFERENCE_SUMMARY = pd.DataFrame([dict(spec=spec, dataset=ds,
                                       pairs_significant_per_fold_holm=int(((w.spec == spec) & (w.dataset == ds) & (w.p_fold_holm < 0.05)).sum()),
                                       pairs_significant_per_project_holm=int(((w.spec == spec) & (w.dataset == ds) & (w.p_project_holm < 0.05)).sum()),
                                       pairs_total=len(PAIRS)) for spec in (PRIMARY, "loglog") for ds in DS_LIST])
RUN_INFO = pd.DataFrame([dict(item=k, value=str(v)) for k, v in {**{f"version[{k}]": v for k, v in VERSIONS.items()}, "platform": platform.platform(),
                         "seeds": SEEDS, "n_splits": N_SPLITS, "primary_spec": PRIMARY, "sd_definition": "sample SD, ddof = 1",
                         "holm_family": "10 pairwise comparisons per dataset", "log1p": "correctly rounded (decimal, 50 digits)", "PRED25_tolerance": PRED_TOL,
                         "np.log1p_not_correctly_rounded_on_this_platform": LOG1P_PLATFORM_DIFF, "run_started_utc": datetime.datetime.utcfromtimestamp(T_START).isoformat(timespec="seconds")}.items()])
README = pd.DataFrame([
    ("T1_datasets", "Table I: projects, features after encoding, effort unit, size features, dropped columns, md5 of the data file"),
    ("T2_main", "Table II: primary protocol (semi-log target, partition seed 42). Mean, sample SD (ddof=1) and median across the 10 folds; *_best flags the best value at the displayed precision (ties flagged together)"),
    ("T3_ablation", "Table III: MAE under three specifications (raw / semilog / loglog). seed42 = primary partition; mean10, sd10 = mean and sample SD over partitions 42-51"),
    ("T4_stability", "Table IV: stability of the primary protocol over the 10 partitions (mean, SD, CV, min, max, position of seed 42)"),
    ("descriptives", "Effort and size descriptive statistics, skewness before and after ln(1+y)"),
    ("per_fold", "dataset x spec x algorithm x fold x 6 metrics (partition seed 42)"),
    ("per_seed", "dataset x spec x algorithm x seed: mean of the 10 folds for 6 metrics"),
    ("friedman", "Friedman chi-square, p, average ranks, CD (per-fold MAE, seed 42) for semilog and loglog"),
    ("nemenyi", "Nemenyi post-hoc p per pair and whether the rank difference exceeds CD"),
    ("wilcoxon_holm", "Wilcoxon signed-rank per fold (n=10) and per project (out-of-fold absolute errors), raw and Holm-adjusted p; effect sizes"),
    ("effect_sizes", "Vargha-Delaney A12 (probability that the first algorithm has the lower absolute error), Cliff's delta, magnitude, paired win rate"),
    ("inference_summary", "Number of pairs significant after Holm, per fold vs per project"),
    ("ablation", "All six metrics by specification: seed42, mean10, sd10, min10, max10 (long format)"),
    ("shap_importance", "Mean |SHAP| per feature (Random Forest on ln(1+effort)), share, Spearman correlation feature value vs SHAP value"),
    ("rank_summary", "Mean MAE rank of each algorithm over the 10 partitions, per specification"),
    ("rank_18_cells", "Mean rank over 3 datasets x 6 metrics (primary protocol): rank-based definition of consistency"),
    ("lr_case", "Projects in the fold where Linear Regression (semi-log) fails worst: size, actual effort, LR predictions under the three specifications"),
    ("lr_worst_by_seed", "Per dataset, specification and partition: LR mean MAE, worst fold MAE, and the project with the largest LR error"),
    ("run_info", "Library versions, seeds, definitions used in this run"),
], columns=["sheet", "content"])
SHEETS = {"README": README, "T1_datasets": T1_DATASETS, "T2_main": T2_MAIN, "T3_ablation": T3_ABLATION, "T4_stability": T4_STABILITY, "descriptives": DESCRIPTIVES,
          "per_fold": PER_FOLD, "per_seed": PER_SEED, "friedman": FRIEDMAN, "nemenyi": NEMENYI, "wilcoxon_holm": WILCOXON_HOLM, "effect_sizes": EFFECT_SIZES,
          "inference_summary": INFERENCE_SUMMARY, "ablation": ABLATION, "shap_importance": SHAP_IMPORTANCE, "rank_summary": RANK_SUMMARY, "rank_18_cells": RANK_18,
          "lr_case": LR_CASE, "lr_worst_by_seed": LR_WORST_DF, "run_info": RUN_INFO}
XLSX = os.path.join(RES, "SEE_results_rev3.xlsx")
with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
    for name, df in SHEETS.items():
        df.to_excel(xw, sheet_name=name, index=False)
        df.to_csv(os.path.join(CSV, f"{name}.csv"), index=False)

# ---- Number trace: manuscript number -> workbook cell
def _col_letter(n):
    s = ""
    while n: n, r = divmod(n - 1, 26); s = chr(65 + r) + s
    return s
def cell(sheet, column, **where):
    df = SHEETS[sheet]; mask = np.ones(len(df), bool)
    for k, v in where.items(): mask &= (df[k] == v).values
    i = int(np.where(mask)[0][0])
    return f"{sheet}!{_col_letter(list(df.columns).index(column) + 1)}{i + 2}", df[column].iloc[i]
TRACE_ITEMS = [
    ("LR mean MAE, COCOMO81 (primary protocol)", cell("T2_main", "MAE_mean", dataset="COCOMO81", algorithm="LR")),
    ("LR across-fold median MAE, COCOMO81", cell("T2_main", "MAE_median_fold", dataset="COCOMO81", algorithm="LR")),
    ("LR mean SA, COCOMO81", cell("T2_main", "SA_mean", dataset="COCOMO81", algorithm="LR")),
    ("LR MMRE / MdMRE, COCOMO81", cell("T2_main", "MMRE_mean", dataset="COCOMO81", algorithm="LR")),
    ("RF MMRE, COCOMO81", cell("T2_main", "MMRE_mean", dataset="COCOMO81", algorithm="RF")),
    ("RF SA, COCOMO81", cell("T2_main", "SA_mean", dataset="COCOMO81", algorithm="RF")),
    ("RF MMRE, NASA93", cell("T2_main", "MMRE_mean", dataset="NASA93", algorithm="RF")),
    ("RF SA, NASA93", cell("T2_main", "SA_mean", dataset="NASA93", algorithm="RF")),
    ("Best PRED(25) of the study (Desharnais, LR)", cell("T2_main", "PRED25_mean", dataset="Desharnais", algorithm="LR")),
    ("MAE of the failing fold, LR semi-log, COCOMO81", cell("lr_case", "fold_MAE_LR_semilog", dataset="COCOMO81")),
    ("Largest project size, COCOMO81 (KLOC)", cell("descriptives", "size_max[loc]", dataset="COCOMO81")),
    ("Skewness of effort, COCOMO81", cell("descriptives", "skewness", dataset="COCOMO81")),
    ("Skewness of ln(1+effort), COCOMO81", cell("descriptives", "skewness_log1p", dataset="COCOMO81")),
    ("LR MAE over 10 partitions: mean, COCOMO81", cell("T4_stability", "MAE_mean10", dataset="COCOMO81", algorithm="LR")),
    ("LR MAE over 10 partitions: min, COCOMO81", cell("T4_stability", "MAE_min10", dataset="COCOMO81", algorithm="LR")),
    ("LR MAE over 10 partitions: max, COCOMO81", cell("T4_stability", "MAE_max10", dataset="COCOMO81", algorithm="LR")),
    ("RF MAE over 10 partitions: mean, COCOMO81", cell("T4_stability", "MAE_mean10", dataset="COCOMO81", algorithm="RF")),
    ("RF MAE over 10 partitions: SD, COCOMO81", cell("T4_stability", "MAE_sd10", dataset="COCOMO81", algorithm="RF")),
    ("LR MAE over 10 partitions: mean, NASA93", cell("T4_stability", "MAE_mean10", dataset="NASA93", algorithm="LR")),
    ("Position of seed 42 among partitions, LR NASA93 (1 = lowest MAE)", cell("T4_stability", "seed42_position_low_to_high", dataset="NASA93", algorithm="LR")),
    ("LR MAE, raw specification, mean of 10 partitions, COCOMO81", cell("T3_ablation", "raw_mean10", dataset="COCOMO81", algorithm="LR")),
    ("LR MAE, log-log specification, mean of 10 partitions, COCOMO81", cell("T3_ablation", "loglog_mean10", dataset="COCOMO81", algorithm="LR")),
    ("LR MAE, log-log specification, mean of 10 partitions, NASA93", cell("T3_ablation", "loglog_mean10", dataset="NASA93", algorithm="LR")),
    ("LR MAE, log-log specification, mean of 10 partitions, Desharnais", cell("T3_ablation", "loglog_mean10", dataset="Desharnais", algorithm="LR")),
    ("Friedman p, COCOMO81, semi-log", cell("friedman", "friedman_p", spec="semilog", dataset="COCOMO81")),
    ("Friedman p, NASA93, semi-log", cell("friedman", "friedman_p", spec="semilog", dataset="NASA93")),
    ("Friedman p, Desharnais, semi-log", cell("friedman", "friedman_p", spec="semilog", dataset="Desharnais")),
    ("Friedman p, COCOMO81, log-log", cell("friedman", "friedman_p", spec="loglog", dataset="COCOMO81")),
    ("Nemenyi p, RF-KNN, COCOMO81, semi-log", cell("nemenyi", "nemenyi_p", spec="semilog", dataset="COCOMO81", pair="RF-KNN")),
    ("Wilcoxon per fold, LR-RF, COCOMO81: raw p", cell("wilcoxon_holm", "p_fold_raw", spec="semilog", dataset="COCOMO81", pair="LR-RF")),
    ("Wilcoxon per fold, LR-RF, COCOMO81: Holm p", cell("wilcoxon_holm", "p_fold_holm", spec="semilog", dataset="COCOMO81", pair="LR-RF")),
    ("Pairs significant after Holm, per project, COCOMO81, semi-log", cell("inference_summary", "pairs_significant_per_project_holm", spec="semilog", dataset="COCOMO81")),
]
if len(SHAP_IMPORTANCE):
    TRACE_ITEMS += [(f"Top SHAP feature share (%), {ds}", cell("shap_importance", "share_percent", dataset=ds, rank=1)) for ds in DS_LIST]
with open("NUMBER_TRACE.md", "w", encoding="utf-8") as fh:
    fh.write("# NUMBER_TRACE (preliminary; completed together with the manuscript in Phase 3)\n\n"
             "Source workbook: `results/SEE_results_rev3.xlsx` (the same tables are in `results/csv/<sheet>.csv`).\n\n"
             "## Table-level mapping\n"
             "- Table I  <- sheet `T1_datasets` (columns projects_raw, projects_used, features, effort_unit)\n"
             "- Table II <- sheet `T2_main`, row = (dataset, algorithm), columns `<metric>_mean` and `<metric>_sd`; bold = `<metric>_best`\n"
             "- Table III <- sheet `T3_ablation`, columns `<spec>_mean10`, `<spec>_sd10` (and `<spec>_seed42`)\n"
             "- Table IV <- sheet `T4_stability`\n"
             "- Inference tables <- sheets `friedman`, `nemenyi`, `wilcoxon_holm`, `effect_sizes`, `inference_summary`\n"
             "- SHAP <- sheet `shap_importance`\n\n## Key in-text numbers\n\n| Number in the manuscript | Cell | Value in this run |\n|---|---|---|\n")
    for label, (address, value) in TRACE_ITEMS:
        fh.write(f"| {label} | `{address}` | {value:,.6g} |\n" if isinstance(value, (int, float, np.integer, np.floating)) else f"| {label} | `{address}` | {value} |\n")

# ---- Self-check against the reference run
def _sig(v):
    v = float(v)
    return "nan" if np.isnan(v) else f"{v:.6g}"
def _digest(frames):
    vals = []
    for df in frames:
        vals += [_sig(v) for v in df.select_dtypes(include=[np.number]).values.ravel()]
    return hashlib.sha256(json.dumps(vals).encode()).hexdigest()[:16]
DIGEST_CORE = _digest([PER_SEED, PER_FOLD, FRIEDMAN, NEMENYI, WILCOXON_HOLM])
DIGEST_SHAP = _digest([SHAP_IMPORTANCE]) if len(SHAP_IMPORTANCE) else None
HEADLINE = {label: float(value) for label, (address, value) in TRACE_ITEMS}
json.dump(dict(digest_core=DIGEST_CORE, digest_shap=DIGEST_SHAP, headline=HEADLINE, versions=VERSIONS), open(os.path.join(RES, "self_check.json"), "w"), indent=1)
REFERENCE = json.loads(r'''{"digest_core": "2ce714e5dadd0252", "digest_shap": "a025890a6bf76878", "headline": {"LR mean MAE, COCOMO81 (primary protocol)": 63492.01524546913, "LR across-fold median MAE, COCOMO81": 315.75600585102615, "LR mean SA, COCOMO81": -3395.4772853751847, "LR MMRE / MdMRE, COCOMO81": 11.432886361700273, "RF MMRE, COCOMO81": 0.7557106147090689, "RF SA, COCOMO81": 65.1924457499928, "RF MMRE, NASA93": 0.6395690723941517, "RF SA, NASA93": 65.83731678440209, "Best PRED(25) of the study (Desharnais, LR)": 40.53571428571429, "MAE of the failing fold, LR semi-log, COCOMO81": 630800.5147923328, "Largest project size, COCOMO81 (KLOC)": 1150.0, "Skewness of effort, COCOMO81": 4.36784332222379, "Skewness of ln(1+effort), COCOMO81": 0.38815197653921735, "LR MAE over 10 partitions: mean, COCOMO81": 195566.6927181025, "LR MAE over 10 partitions: min, COCOMO81": 39384.24539675502, "LR MAE over 10 partitions: max, COCOMO81": 995510.1785042544, "RF MAE over 10 partitions: mean, COCOMO81": 442.83133973070187, "RF MAE over 10 partitions: SD, COCOMO81": 13.404066959778813, "LR MAE over 10 partitions: mean, NASA93": 1897.9539213762496, "Position of seed 42 among partitions, LR NASA93 (1 = lowest MAE)": 1.0, "LR MAE, raw specification, mean of 10 partitions, COCOMO81": 981.664909147582, "LR MAE, log-log specification, mean of 10 partitions, COCOMO81": 282.03676231659495, "LR MAE, log-log specification, mean of 10 partitions, NASA93": 254.69377397640764, "LR MAE, log-log specification, mean of 10 partitions, Desharnais": 1809.1202746157985, "Friedman p, COCOMO81, semi-log": 0.0378170468169309, "Friedman p, NASA93, semi-log": 0.05270479887852743, "Friedman p, Desharnais, semi-log": 0.01033879708536391, "Friedman p, COCOMO81, log-log": 0.0004014124299719765, "Nemenyi p, RF-KNN, COCOMO81, semi-log": 0.03769606822054161, "Wilcoxon per fold, LR-RF, COCOMO81: raw p": 0.00390625, "Wilcoxon per fold, LR-RF, COCOMO81: Holm p": 0.0390625, "Pairs significant after Holm, per project, COCOMO81, semi-log": 3.0, "Top SHAP feature share (%), COCOMO81": 58.089609780442, "Top SHAP feature share (%), NASA93": 68.54353734197618, "Top SHAP feature share (%), Desharnais": 36.714761642914546}}''')
CHECK_LINES = [f"digest_core = {DIGEST_CORE}", f"digest_shap = {DIGEST_SHAP}"]
if REFERENCE is None:
    CHECK_LINES.append("SELF-CHECK: no reference embedded (this is the reference run)")
else:
    bad = [(k, v, REFERENCE["headline"][k]) for k, v in HEADLINE.items() if k in REFERENCE["headline"] and abs(v - REFERENCE["headline"][k]) > 1e-6 * max(1.0, abs(REFERENCE["headline"][k]))]
    ok_core = DIGEST_CORE == REFERENCE["digest_core"]; ok_shap = (DIGEST_SHAP is None) or (DIGEST_SHAP == REFERENCE["digest_shap"])
    CHECK_LINES.append(f"SELF-CHECK core numbers (per_seed, per_fold, inference): {'PASS' if ok_core else 'DIFFERENT from the reference run'}")
    CHECK_LINES.append(f"SELF-CHECK SHAP importance: {'PASS' if ok_shap else 'DIFFERENT from the reference run'}")
    CHECK_LINES.append(f"SELF-CHECK headline numbers: {len(HEADLINE) - len(bad)}/{len(HEADLINE)} identical (relative tolerance 1e-6)")
    CHECK_LINES += [f"   DIFFERENT: {k}: this run {v:.8g}, reference {r:.8g}" for k, v, r in bad]

with open("run_log.txt", "w", encoding="utf-8") as fh:
    fh.write("SEE_IJCS_rev3 run log\n")
    fh.write(f"run started (UTC): {datetime.datetime.utcfromtimestamp(T_START).isoformat(timespec='seconds')}\n")
    fh.write(f"run finished (UTC): {datetime.datetime.utcnow().isoformat(timespec='seconds')}\nruntime: {time.time() - T_START:.0f} s\n")
    fh.write(f"platform: {platform.platform()}\n" + "".join(f"{k}: {v}\n" for k, v in VERSIONS.items()))
    fh.write("".join(f"[pin] {l}\n" for l in PIN_LOG) + "".join(f"[version difference] {l}\n" for l in VERSION_NOTES))
    fh.write(f"seeds: {SEEDS}; folds: {N_SPLITS}; primary specification: {PRIMARY}; SD: sample (ddof=1)\n")
    for _, r in T1_DATASETS.iterrows():
        fh.write(f"md5 {r['md5']}  {DATASETS[r['dataset']]['file']}  (matches reference: {r['md5_matches_reference']})\n")
    fh.write(f"targets where np.log1p of this platform is not correctly rounded (handled by log1p_exact): {LOG1P_PLATFORM_DIFF}\n")
    fh.write("\n".join(CHECK_LINES) + "\n" + f"figures: {', '.join(FIGURE_FILES) if FIGURE_FILES else 'skipped'}\n")
print("\n".join(CHECK_LINES)); print(f"workbook: {XLSX} | sheets: {len(SHEETS)} | runtime {time.time() - T_START:.0f} s")

# =============================================================================
# 10. PACK THE OUTPUTS (and offer the download when running in Google Colab)
# =============================================================================
import zipfile
ZIP_NAME = "SEE_rev3_outputs.zip"
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in (RES, FIG):
        for folder, _, files in os.walk(root):
            for f in sorted(files):
                zf.write(os.path.join(folder, f))
    for f in ("run_log.txt", "NUMBER_TRACE.md"):
        zf.write(f)
print(f"{ZIP_NAME}: {os.path.getsize(ZIP_NAME) / 1e6:.1f} MB")
try:
    from google.colab import files as _colab_files
    _colab_files.download(ZIP_NAME)
except Exception:
    print("Not running in Colab: the archive is in the working directory.")