"""Checks every qualitative statement of the sensitivity analysis against results_rev4/*.csv. Usage: python check_claims.py <results dir>"""
import sys, os, pandas as pd
r = sys.argv[1]
S = pd.read_csv(os.path.join(r, "summary_rev4.csv")); C = pd.read_csv(os.path.join(r, "tuned_vs_default.csv")); W = pd.read_csv(os.path.join(r, "wins_tuned.csv"))
m = lambda ds, sp, a, arm: float(S[(S.dataset == ds) & (S.spec == sp) & (S.algorithm == a) & (S.arm == arm)].MAE_mean10.iloc[0])
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name)
DS = ["COCOMO81", "NASA93", "Desharnais"]; OTH = ["DT", "RF", "KNN", "SVR", "GB"]
chk("tuned RF higher than default RF in all six combinations", (C[C.algorithm == "RF"].change_percent > 0).all())
chk("non-SVR change at most 10.8%", C[C.algorithm != "SVR"].change_percent.abs().max() <= 10.85)
chk("SVR largest decrease is COCOMO81 log-log", tuple(C.loc[C[C.algorithm == "SVR"].change_percent.idxmin(), ["dataset", "spec"]]) == ("COCOMO81", "loglog"))
chk("SVR largest increase is NASA93 semi-log", tuple(C.loc[C[C.algorithm == "SVR"].change_percent.idxmax(), ["dataset", "spec"]]) == ("NASA93", "semilog"))
for ds in ("COCOMO81", "NASA93"):
    chk(f"GB default < RF default, semi-log, {ds}", m(ds, "semilog", "GB", "default") < m(ds, "semilog", "RF", "default"))
    chk(f"GB default < RF default, log-log, {ds}", m(ds, "loglog", "GB", "default") < m(ds, "loglog", "RF", "default"))
    chk(f"GB lowest of six (default and tuned), semi-log, {ds}", all(m(ds, "semilog", "GB", arm) <= min([m(ds, "semilog", a, arm) for a in OTH] + [m(ds, "semilog", "LR", "default")]) for arm in ("default", "tuned")))
for ds in ("COCOMO81", "Desharnais"):
    chk(f"log-log LR lowest after tuning, {ds}", m(ds, "loglog", "LR", "default") < min(m(ds, "loglog", a, arm) for a in OTH for arm in ("default", "tuned")))
lr = m("NASA93", "loglog", "LR", "default")
below = [(a, arm) for a in OTH for arm in ("default", "tuned") if m("NASA93", "loglog", a, arm) < lr]
chk("NASA93 log-log: only tuned SVR below LR", below == [("SVR", "tuned")])
chk("GB never below log-log LR", all(m(ds, "loglog", "GB", arm) > m(ds, "loglog", "LR", "default") for ds in DS for arm in ("default", "tuned")))
chk("log-log LR lowest of six with default settings on all datasets", all(m(ds, "loglog", "LR", "default") < min(m(ds, "loglog", a, "default") for a in OTH) for ds in DS))
b = S.loc[S.PRED25_mean10.idxmax()]
chk("best PRED(25) is tuned SVR, log-log, NASA93 and below 75", (b.dataset, b.spec, b.algorithm, b.arm) == ("NASA93", "loglog", "SVR", "tuned") and b.PRED25_mean10 < 75)
chk("tuned SVR close to LR on COCOMO81 log-log (within 5%)", abs(m("COCOMO81", "loglog", "SVR", "tuned") / m("COCOMO81", "loglog", "LR", "default") - 1) < 0.05)
print("ALL PASS" if ok else "SOME CLAIMS FAIL"); sys.exit(0 if ok else 1)
