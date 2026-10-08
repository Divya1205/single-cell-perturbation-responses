"""
IFNβ / IFNγ response in PBMCs (Dong et al. 2023)

Questions
  1. Which cell types respond to IFNβ, IFNγ and the combination, and how strongly?
  2. Is the combined response additive, or larger or smaller than the sum of the single responses?

Focused test: does IFNβ reduce IFNγ's effect on MHC class II in monocytes?
  - IFNγ is the classic inducer of MHC-II (through CIITA); monocytes are the PBMCs that both
    express MHC-II and respond strongly to IFNγ.
  - Earlier studies reported that IFNβ blocks IFNγ-induced MHC-II in mouse macrophages and
    other cell types (Ling et al. 1985; Inaba et al. 1986; Lu et al. 1995).

Multiple testing: Benjamini–Hochberg FDR across exploratory comparisons. The MHC-II test is the
literature-motivated hypothesis; all other results (including CXCL9) are exploratory.

Input : data/dong_2023_raw.h5ad (downloaded with pertpy.data.dong_2023() on first run)
Output: tables in results/, figures in results/figures/
Run   : python analysis.py   (section 2b downloads the MSigDB Hallmark sets; needs internet)
"""

# %% Setup
import re
import warnings

import matplotlib
matplotlib.use("Agg")                       # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", message="IProgress not found")
SEED = 0                                    # each section re-seeds from this, so results are reproducible
RES = Path("results"); FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def save_fig(name, **kw):
    plt.savefig(FIG / name, dpi=150, **kw)
    plt.close("all")
    print(f"[saved figure] {FIG / name}")


# %% 1. Load the data
# X in this file holds only 773 scaled genes, so we use adata.raw:
# log-normalised expression of all 21,819 genes (already in the file).
DATA = Path("data/dong_2023_raw.h5ad")
if DATA.exists():
    adata = sc.read_h5ad(DATA)
else:                                       # first run: download through pertpy and keep a local copy
    import pertpy as pt
    adata = pt.dt.dong_2023()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(DATA)
ad = adata.raw.to_adata()                   # all genes, log-normalised
ad.obsm["X_umap"] = adata.obsm["X_umap"]    # keep the authors' UMAP
print(ad)
print(f"\nX (scaled): {adata.n_vars} genes | raw: {ad.n_vars} genes")
print("raw min / max:", float(ad.X.min()), "/", float(ad.X.max()))

# Is .raw normalised? After undoing the log, every cell should have the same total.
tot = np.asarray(ad.X.expm1().sum(1)).ravel()
print("expm1(raw) per-cell totals 5/50/95%:", np.percentile(tot, [5, 50, 95]).round(0))
if "total_counts" in ad.obs:
    print("original total_counts 5/50/95%:   ", np.percentile(ad.obs["total_counts"], [5, 50, 95]).round(0))
    print("correlation with original depth:  ", np.corrcoef(tot, ad.obs["total_counts"])[0, 1].round(3))
print("-> same total for every cell = normalised, then log1p.")

# Conditions and cell types
COND = {"No stimulation": "ctrl", "IFNb": "IFNb", "IFNg": "IFNg", "IFNb+ IFNg": "IFNb+IFNg"}
ORDER = list(COND.values())
ad.obs["condition"] = pd.Categorical(ad.obs["perturbation"].astype(str).map(COND), categories=ORDER)
ad.obs["cell_type"] = ad.obs["cell_type0528"].astype(str)
cond = ad.obs["condition"].astype(str).values
ct = ad.obs["cell_type"].values

tab = pd.crosstab(ad.obs["cell_type"], ad.obs["condition"])
TYPES = tab.index[(tab >= 20).all(axis=1)].tolist()
print("\n", tab, "\n\nCell types with >= 20 cells in every condition (analysed):", TYPES)
for b in ad.obs["batch"].astype(str).unique():
    m = re.fullmatch(r"H(\d+)D(\d+)", b)
    label = f"donor {m.group(1)}, day {m.group(2)} of stimulation" if m else "unrecognised format"
    print(f"Batch {b}: {label}")
print("-> one donor: cells are the only replicates")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(ad, color="condition", ax=axes[0], show=False)
sc.pl.umap(ad, color="cell_type", ax=axes[1], show=False)
plt.tight_layout(); save_fig("nb_umap.png")


# %% 2a. Gene-set scores
# Gene sets taken from established interferon-stimulated and antigen-presentation genes,
# not selected from this data. Score per cell = mean log expression of the set's genes.
#   type_I_ISG    : response to IFNβ
#   IFNG_specific : response to IFNγ
#   MHC_II        : antigen presentation to helper T cells (focused test)
#   MHC_I         : antigen presentation to killer T cells
GENE_SETS = {
    "type_I_ISG":    ["ISG15", "MX1", "MX2", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI44L",
                      "OAS1", "OAS2", "OAS3", "RSAD2", "HERC5"],
    "IFNG_specific": ["CXCL9", "CIITA", "IDO1", "GBP1", "GBP4", "GBP5"],
    "MHC_II":        ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
                      "HLA-DMA", "HLA-DMB", "CD74"],
    "MHC_I":         ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M"],
}
SETS = {}
print()
for name, genes in GENE_SETS.items():
    present = [g for g in genes if g in ad.var_names]
    missing = sorted(set(genes) - set(present))
    print(f"{name:<14} {len(present)}/{len(genes)} genes present" + (f" (missing: {missing})" if missing else ""))
    if present:
        SETS[name] = present
        ad.obs[name] = np.asarray(ad[:, present].X.mean(axis=1)).ravel()

# Mean score per cell type and condition (each panel has its own colour scale)
sub = ad.obs[ad.obs.cell_type.isin(TYPES)]
fig, axes = plt.subplots(1, len(SETS), figsize=(4.2 * len(SETS), 3.6))
for ax, name in zip(np.atleast_1d(axes), SETS):
    piv = sub.pivot_table(index="cell_type", columns="condition", values=name, observed=True)[ORDER]
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(4), ORDER, rotation=30); ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(name); plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); save_fig("nb_set_scores.png")


# %% 2b. Check the gene sets against MSigDB Hallmark (Liberzon et al. 2015)
import decoupler as dc

hm = dc.op.hallmark(organism="human")       # downloads once
alpha_src = [s for s in hm.source.unique() if "INTERFERON_ALPHA" in s.upper()][0]
gamma_src = [s for s in hm.source.unique() if "INTERFERON_GAMMA" in s.upper()][0]
alpha = set(hm[hm.source == alpha_src].target)
gamma = set(hm[hm.source == gamma_src].target)

print()
for mine, ref, src in [("type_I_ISG", alpha, alpha_src), ("IFNG_specific", gamma, gamma_src)]:
    print(f"{mine}: {sum(g in ref for g in SETS[mine])}/{len(SETS[mine])} genes in {src}")
for name in ["type_I_ISG", "IFNG_specific"]:
    for g in SETS[name]:
        where = ", ".join(x for x, s in [("alpha", alpha), ("gamma", gamma)] if g in s) or "neither"
        print(f"{name:<14} {g:<8} -> Hallmark {where}")
# IFI6 and HERC5 (type I) and GBP1/GBP5 (IFNγ) are in neither set but are documented
# interferon-induced genes (Schoggins et al. 2011; Wong et al. 2006; Tretina et al. 2019).


# %% 2c. Where are these genes active? (UMAP)
show_these = [g for g in ["ISG15", "CXCL9", "HLA-DRA", "CIITA", "type_I_ISG", "IFNG_specific", "MHC_II"]
              if g in ad.var_names or g in ad.obs]
sc.pl.umap(ad, color=show_these, ncols=4, cmap="viridis", vmax="p99", show=False)
save_fig("nb_umap_genes.png", bbox_inches="tight")

# Dendritic cells rise from 4 (ctrl) to 10-17 (stimulated). On the UMAP they form their own
# cluster, separate from the monocytes. A formal pooling check is listed as future work.
m = ad.obs["cell_type"] == "Dendritic"
ax = sc.pl.umap(ad, color="condition", show=False, title="Dendritic cells (circled)")
ax.scatter(*ad.obsm["X_umap"][m.values].T, s=40, facecolors="none", edgecolors="k", lw=0.8, label="Dendritic")
ax.legend(loc="lower left", fontsize=7)
save_fig("nb_umap_dendritic.png", bbox_inches="tight")


# %% 3. Which cell types respond?
# Two readouts per cell type and stimulus, both at equal cell numbers, so differences in
# cell-type abundance do not bias the ranking. Features: the authors' 773 highly variable genes.
#   a) Classifier AUC (approach of Augur; Skinnider et al. 2021): logistic regression,
#      20 stimulated vs 20 control cells, 3-fold CV, 30 repeats. 0.5 = no response, 1 = separable.
#      "ctrl vs ctrl" (two random halves of control cells) should be about 0.5.
#   b) Signal-to-noise ratio: distance between average stimulated and average control profile,
#      divided by the distance between two random halves of control cells. 1 = no bigger than noise.
rng = np.random.default_rng(SEED)
feat = [g for g in adata.var_names if g in ad.var_names]
Xf = ad[:, feat].X
Xf = Xf.toarray() if hasattr(Xf, "toarray") else np.asarray(Xf)
N, REPEATS = 20, 30
STIMS = ["IFNb", "IFNg", "IFNb+IFNg"]


def auc_once(a, b):
    Xs, y = np.vstack([Xf[a], Xf[b]]), np.r_[np.zeros(len(a)), np.ones(len(b))]
    cv = StratifiedKFold(3, shuffle=True, random_state=int(rng.integers(1e9)))
    return cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=cv, scoring="roc_auc").mean()


rows = []
for t in TYPES:
    ic = np.where((ct == t) & (cond == "ctrl"))[0]
    aucs = []                                   # control-vs-control check
    for _ in range(REPEATS):
        p = rng.permutation(ic)
        aucs.append(auc_once(p[:N], p[N:2 * N]))
    rows.append({"cell_type": t, "stim": "ctrl vs ctrl", "auc": np.mean(aucs), "sd": np.std(aucs)})
    for s in STIMS:
        is_ = np.where((ct == t) & (cond == s))[0]
        aucs = [auc_once(rng.choice(ic, N, replace=False), rng.choice(is_, N, replace=False))
                for _ in range(REPEATS)]
        rows.append({"cell_type": t, "stim": s, "auc": np.mean(aucs), "sd": np.std(aucs)})
auc = pd.DataFrame(rows)
auc_wide = auc.pivot(index="cell_type", columns="stim", values="auc")[["ctrl vs ctrl"] + STIMS]
print("\nClassifier AUC (stimulated vs control):\n", auc_wide.round(3))
auc_wide.round(3).to_csv(RES / "nb_auc.csv")

fig, ax = plt.subplots(figsize=(6, 0.5 * len(auc_wide) + 1.5))
im = ax.imshow(auc_wide.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=1)
ax.set_xticks(range(auc_wide.shape[1]), auc_wide.columns, rotation=20)
ax.set_yticks(range(len(auc_wide)), auc_wide.index)
for i in range(auc_wide.shape[0]):
    for j in range(auc_wide.shape[1]):
        ax.text(j, i, f"{auc_wide.values[i, j]:.2f}", ha="center", va="center", color="w")
plt.colorbar(im, label="AUC"); ax.set_title("Response strength (classifier AUC)")
plt.tight_layout(); save_fig("nb_response_auc.png")

# IFNβ AUCs are all near 1, so the AUC cannot rank IFNβ responses; the signal-to-noise ratio can.
rng = np.random.default_rng(SEED + 1)


def magnitude(t, s, n=40, reps=20):
    ic = np.where((ct == t) & (cond == "ctrl"))[0]
    is_ = np.where((ct == t) & (cond == s))[0]
    n = min(n, len(ic) // 2, len(is_))
    sig, noise = [], []
    for _ in range(reps):
        p = rng.permutation(ic)
        c1, c2 = p[:n], p[n:2 * n]
        st = rng.choice(is_, n, replace=False)
        sig.append(np.linalg.norm(Xf[st].mean(0) - Xf[c1].mean(0)))
        noise.append(np.linalg.norm(Xf[c2].mean(0) - Xf[c1].mean(0)))
    return np.mean(sig) / np.mean(noise)


mag = pd.DataFrame({s: {t: magnitude(t, s) for t in TYPES} for s in STIMS})[STIMS]
print("\nSignal-to-noise ratio (1 = no bigger than noise):\n", mag.round(2))
mag.round(2).to_csv(RES / "nb_signal_to_noise.csv")


# %% 4. Does IFNβ reduce IFNγ-driven MHC-II in monocytes?
# Comparisons (from the mean scores of ctrl, IFNβ, IFNγ and combo):
#   IFNβ effect   = IFNβ - ctrl
#   IFNγ effect   = IFNγ - ctrl
#   combo - IFNγ  : is the combination lower than IFNγ alone?
#   interaction   = (combo - IFNβ) - (IFNγ - ctrl): does IFNγ's effect change when IFNβ is present?
#                   below 0 = IFNβ reduces IFNγ's effect; about 0 = the effects add.
# combo - IFNγ alone can mislead: IFNβ lowers MHC-II on its own, so the combination can be lower
# than IFNγ alone even when IFNγ's effect is unchanged.
# Statistics: two-way ANOVA (score ~ IFNβ × IFNγ) on cells, HC3 robust 95% CIs (MacKinnon & White 1985);
# "up/down" = CI excludes 0; q = Benjamini–Hochberg-adjusted p (Benjamini & Hochberg 1995).
CONTRASTS = ["IFNb_effect", "IFNg_effect", "combo_minus_IFNg", "interaction"]
# Each comparison as weights on the model coefficients (Intercept = ctrl mean)
WEIGHTS = {"IFNb_effect":      {"IFNb": 1},
           "IFNg_effect":      {"IFNg": 1},
           "combo_minus_IFNg": {"IFNb": 1, "IFNb:IFNg": 1},
           "interaction":      {"IFNb:IFNg": 1},
           "combo_effect":     {"IFNb": 1, "IFNg": 1, "IFNb:IFNg": 1},   # combo - ctrl (section 5)
           "IFNg_with_IFNb":   {"IFNg": 1, "IFNb:IFNg": 1}}              # combo - IFNb (section 4b)


def lm_contrasts(v, c):
    """Two-way linear model on cells (= two-way ANOVA with interaction): y ~ IFNb * IFNg,
    OLS with HC3 standard errors. Returns {comparison: (estimate, ci_low, ci_high, p_value)}."""
    d = pd.DataFrame({"y": np.asarray(v, float),
                      "IFNb": np.isin(c, ["IFNb", "IFNb+IFNg"]).astype(int),
                      "IFNg": np.isin(c, ["IFNg", "IFNb+IFNg"]).astype(int)})
    fit = smf.ols("y ~ IFNb * IFNg", data=d).fit(cov_type="HC3")
    out = {}
    for k, w in WEIGHTS.items():
        t = fit.t_test(np.array([[w.get(n, 0) for n in fit.params.index]], float))
        lo, hi = np.asarray(t.conf_int()).ravel()
        out[k] = (float(np.ravel(t.effect)[0]), float(lo), float(hi), float(np.ravel(t.pvalue)[0]))
    return out


def add_calls(r):
    """Call from the 95% CI; q = Benjamini-Hochberg across the 4 main comparisons of the whole table."""
    r["call"] = np.select([r.hi < 0, r.lo > 0], ["down", "up"], "n.s.")
    main = r.contrast.isin(CONTRASTS)
    r["q"] = np.nan
    r.loc[main, "q"] = multipletests(r.loc[main, "p"], method="fdr_bh")[1]
    fq = lambda q: "" if np.isnan(q) else ("q<0.001" if q < 0.001 else f"q={q:.3f}")
    r["txt"] = r.apply(lambda x: f"{x.est:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}] {x.call} {fq(x.q)}", axis=1)
    return r


def contrast_table(groups):
    rows = []
    for gname, m in groups.items():
        for name in SETS:
            for k, (est, lo, hi, pv) in lm_contrasts(ad.obs[name].values[m], cond[m]).items():
                rows.append({"cell_type": gname, "set": name, "contrast": k,
                             "est": est, "lo": lo, "hi": hi, "p": pv})
    return add_calls(pd.DataFrame(rows))


def verdict(r, group, s="MHC_II"):
    """Plain-language call from three comparisons."""
    d = r[(r.cell_type == group) & (r.set == s)].set_index("contrast")["call"]
    g, cm, it = d["IFNg_effect"], d["combo_minus_IFNg"], d["interaction"]
    if g == "n.s.":
        return f"IFNg has no clear effect on {s} here, so this cannot be judged"
    opp = "down" if g == "up" else "up"
    same_w, opp_w = ("raises", "lowers") if g == "up" else ("lowers", "raises")
    if it == opp:
        return ("IFNb reduces IFNg's effect" if cm == opp
                else f"IFNg's effect is smaller with IFNb, but IFNb {same_w} {s} on its own (ceiling effect)")
    if it == g:
        return "IFNg's effect is larger with IFNb (super-additive)"
    if cm == opp:
        return f"IFNb {opp_w} {s} on its own; no evidence that it reduces IFNg's effect"
    return "no clear change in IFNg's effect (interaction CI includes 0)"


res = contrast_table({t: ct == t for t in TYPES})
res.to_csv(RES / "nb_set_contrasts.csv", index=False)

MONO = [t for t in TYPES if "mono" in t.lower()]
print("\nMonocyte group(s):", MONO)
for t in MONO:
    print(f"FOCUSED TEST   ({t}, MHC-II):        {verdict(res, t)}")
    print(f"Exploratory    ({t}, IFNg-specific): {verdict(res, t, 'IFNG_specific')}")
print("\nMonocytes, gene sets (estimate [95% CI], call, q):")
print(res[res.cell_type.isin(MONO)].pivot(index="set", columns="contrast", values="txt")[CONTRASTS])


# %% 4b. Per gene in monocytes
# The same test for each gene in the MHC-II, IFNγ-specific and MHC-I sets.
#   middle panel: IFNγ's effect without IFNβ (blue circles) and with IFNβ (orange squares);
#                 a long grey line means IFNβ changes IFNγ's effect on that gene.
#   right panel : combination - IFNγ alone. Below 0 with a long grey line = IFNβ blocks IFNγ
#                 (e.g. CXCL9); about 0 = ceiling effect (e.g. GBP5).
#   it_q        : Benjamini-Hochberg-adjusted p for the interaction across these genes.
mono = ad[ad.obs.cell_type.isin(MONO)].copy()
genes = list(dict.fromkeys(g for g in GENE_SETS["MHC_II"] + GENE_SETS["IFNG_specific"] + GENE_SETS["MHC_I"]
                           if g in mono.var_names))
mc = mono.obs["condition"].astype(str).values

rows = []
for g in genes:
    x = mono[:, g].X
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x).ravel()
    r = lm_contrasts(x, mc)
    rows.append({"gene": g, "IFNb_effect": r["IFNb_effect"][0], "IFNg_effect": r["IFNg_effect"][0],
                 "g_lo": r["IFNg_effect"][1], "g_hi": r["IFNg_effect"][2],
                 "w": r["IFNg_with_IFNb"][0], "w_lo": r["IFNg_with_IFNb"][1], "w_hi": r["IFNg_with_IFNb"][2],
                 "combo_minus_IFNg": r["combo_minus_IFNg"][0], "cm_lo": r["combo_minus_IFNg"][1],
                 "cm_hi": r["combo_minus_IFNg"][2],
                 "interaction": r["interaction"][0], "it_lo": r["interaction"][1], "it_hi": r["interaction"][2],
                 "it_p": r["interaction"][3]})
pg = pd.DataFrame(rows)
pg["it_q"] = multipletests(pg["it_p"], method="fdr_bh")[1]
pg = pg.sort_values("interaction")
pg.to_csv(RES / "nb_monocyte_per_gene.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
y = np.arange(len(pg))

# left: MHC-II score per condition
pal = dict(zip(ad.obs["condition"].cat.categories, ad.uns.get("condition_colors", [f"C{i}" for i in range(4)])))
parts = axes[0].violinplot([mono.obs.loc[mc == k, "MHC_II"].values for k in ORDER], showmedians=True)
for body, k in zip(parts["bodies"], ORDER):
    body.set_facecolor(pal[k]); body.set_alpha(0.8)
axes[0].set_xticks(range(1, 5), ORDER); axes[0].set_ylabel("MHC-II score (mean log expr.)")
axes[0].set_title("Monocytes: MHC-II score")

# middle: IFNγ's effect without vs with IFNβ (the gap is the interaction)
ax = axes[1]
ax.hlines(y, pg["IFNg_effect"], pg["w"], color="lightgrey", lw=2)
ax.errorbar(pg["IFNg_effect"], y, xerr=[pg["IFNg_effect"] - pg["g_lo"], pg["g_hi"] - pg["IFNg_effect"]],
            fmt="o", ms=5, label="IFNγ alone (IFNγ − ctrl)")
ax.errorbar(pg["w"], y, xerr=[pg["w"] - pg["w_lo"], pg["w_hi"] - pg["w"]],
            fmt="s", ms=5, label="IFNγ with IFNβ (combo − IFNβ)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8)
ax.set_title("IFNγ's effect without vs with IFNβ")

# right: combination - IFNγ alone
ax = axes[2]
ax.errorbar(pg["combo_minus_IFNg"], y,
            xerr=[pg["combo_minus_IFNg"] - pg["cm_lo"], pg["cm_hi"] - pg["combo_minus_IFNg"]], fmt="o", ms=4)
ax.set_title("combo − IFNγ alone")

for ax in axes[1:]:
    ax.axvline(0, c="k", lw=0.7); ax.set_yticks(y, pg.gene); ax.set_xlabel("log scale, 95% CI")
plt.tight_layout(); save_fig("nb_monocyte_mhc.png")
print("\nMonocytes, per gene (w = IFNγ's effect with IFNβ):")
print(pg[["gene", "IFNb_effect", "IFNg_effect", "w", "combo_minus_IFNg", "interaction", "it_q"]]
      .round(3).to_string(index=False))


# %% 5. Is the combination additive? (all cell types)
# Observed combination effect vs the sum of the two single effects. Error bar = 95% CI of the
# observed combination; the printed table gives the interaction (0 = additive).
for name in ["type_I_ISG", "MHC_II"]:
    d = res[res.set == name].pivot(index="cell_type", columns="contrast", values="est").loc[TYPES]
    combo = d["combo_effect"]
    ce = res[(res.set == name) & (res.contrast == "combo_effect")].set_index("cell_type")
    ci = {t: (ce.loc[t, "lo"], ce.loc[t, "hi"]) for t in TYPES}
    bars = pd.DataFrame({"IFNβ": d["IFNb_effect"], "IFNγ": d["IFNg_effect"],
                         "sum of singles": d["IFNb_effect"] + d["IFNg_effect"], "observed combo": combo})
    fig, ax = plt.subplots(figsize=(8, 4))
    bars.plot.bar(ax=ax, rot=30)
    err = np.array([[combo[t] - ci[t][0], ci[t][1] - combo[t]] for t in TYPES]).T
    ax.errorbar(np.arange(len(TYPES)) + 0.1875, combo.values, yerr=err, fmt="none", ecolor="k", capsize=3)
    ax.axhline(0, c="k", lw=0.7); ax.set_title(f"{name}: change from control (log)"); ax.set_xlabel("")
    plt.tight_layout(); save_fig(f"nb_additivity_{name}.png")
    print(f"\n{name}: interaction per cell type (estimate [95% CI], call, q):")
    print(res[(res.set == name) & (res.contrast == "interaction")][["cell_type", "txt"]]
          .rename(columns={"txt": "interaction"}).set_index("cell_type"))

print("\nDone. Tables in results/, figures in results/figures/.")

# References
#  Dong M et al. Nat Methods 2023;20:1769-1779 (dataset)
#  Ling PD et al. J Immunol 1985;135:1857-1863; Inaba K et al. J Exp Med 1986;163:1030-1035;
#  Lu HT et al. J Exp Med 1995;182:1517-1525 (IFNβ vs IFNγ-induced MHC-II)
#  Liberzon A et al. Cell Syst 2015;1:417-425 (MSigDB Hallmark)
#  Skinnider MA et al. Nat Biotechnol 2021;39:30-34 (Augur)
#  Schoggins JW et al. Nature 2011;472:481-485; Wong JJ et al. PNAS 2006;103:10735-10740;
#  Tretina K et al. J Exp Med 2019;216:482-500 (gene-set genes)
#  Benjamini Y, Hochberg Y. J R Stat Soc B 1995;57:289-300; MacKinnon JG, White H. J Econometrics 1985;29:305-325
