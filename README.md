# Exploring Single-Cell Perturbation Responses
### Interaction of type I and type II interferon signalling in human PBMCs

## Question

1. Which immune cell types respond to IFN-β, IFN-γ and their combination, and how strongly?
2. Is the combined response additive, or larger or smaller than the sum of the two single responses?

**Focused test:** does IFN-β dampen IFN-γ-driven MHC class II expression in monocytes? This is the pre-specified hypothesis; all other comparisons are exploratory.

Why this test: IFN-γ is the classic inducer of MHC-II (through CIITA), and among PBMCs monocytes both express MHC-II and respond strongly to IFN-γ. Earlier studies reported that IFN-β blocks IFN-γ-induced MHC-II in mouse macrophages and other cell types (Ling et al. 1985; Inaba et al. 1986; Lu et al. 1995).

## Data

Dong, M. et al. *Causal identification of single-cell experimental perturbation effects with CINEMA-OT.* Nature Methods (2023).

The scRNA-seq data of human PBMCs under single and combinatorial cytokine stimulation is loaded through pertpy (`pertpy.data.dong_2023()`). The analysis uses the authors' processed data as provided: log-normalised expression of all 21,819 genes (`adata.raw`), their cell-type annotation and their UMAP. No reprocessing is done.

**Subset used:**
- **Conditions:** unstimulated control, IFN-β, IFN-γ, IFN-β + IFN-γ.
- **Cell types:** B, CD4 T, CD8 T, monocytes and NK cells, i.e. those with at least 20 cells in every condition. Dendritic and plasma cells are excluded.
- **Donor:** all cells come from one donor (batch H3D2), so cells are the only replicates.

## Approach

1. **Gene-set scores.** Mean log expression per cell for type I ISGs, IFN-γ-specific genes, MHC-I and MHC-II. The sets were chosen from the literature, not from this data, and are cross-checked against the MSigDB Hallmark interferon-α/γ sets.
2. **Response strength per cell type**, both at equal cell numbers:
   - Classifier AUC (Augur-style): 20 vs 20 cells, 3-fold CV, 30 repeats, on the authors' 773 highly variable genes. A ctrl-vs-ctrl control is included.
   - Signal-to-noise ratio: the shift in the mean profile relative to the difference between two random halves of control cells.
3. **Additivity.** Two-way linear model per cell type, `score ~ IFNβ × IFNγ`, with robust (HC3) 95% CIs. The interaction term measures departure from additivity. Benjamini-Hochberg correction is applied across exploratory comparisons.
4. **Per-gene test in monocytes.** The same model is fitted for each MHC-II, IFN-γ-specific and MHC-I gene.

## Main results

- **Response strength.** IFN-β triggers a strong response in all five cell types. IFN-γ acts mainly on monocytes, and partly on B cells. Monocytes show the largest response to every stimulus.
- **MHC-II in monocytes (focused test).** IFN-γ raises MHC-II and IFN-β lowers it on its own. The interaction is ≈ 0 (+0.06, 95% CI −0.01 to +0.13), so the effects add and there is **no dampening**.
- **IFN-γ-specific genes in monocytes.** IFN-β clearly dampens the IFN-γ response (interaction −0.41). CXCL9 is the strongest case: the combination is lower than IFN-γ alone.
- **Type I ISGs and MHC-I in monocytes.** The response is sub-additive: the combination is below the sum of the singles but at least as high as the stronger single stimulus, consistent with saturation.

## Limitations

All cells come from a single donor, so CIs and p-values reflect cell-to-cell variation only and overstate certainty about donor-level effects. Gene-set scores are unweighted means of log expression.

## Setup

```bash
git clone https://github.com/Divya1205/exploring-single-cell-perturbation-responses.git
cd exploring-single-cell-perturbation-responses
conda env create -f environment.yml
conda activate perturb
```

## Usage

The analysis is available in two equivalent forms. Run either one from the repository root.

**Script.** This form is non-interactive and downloads the data automatically on the first run:

```bash
python analysis.py
```

Results are printed to the terminal. Tables are saved to `results/` and figures to `results/figures/`.

**Notebook.** This form shows the same analysis with the outputs inline:

```bash
python download_data.py     # saves data/dong_2023_raw.h5ad
jupyter lab analysis.ipynb  # then Kernel -> Restart & Run All
```

Both download the MSigDB Hallmark gene sets in section 2b, so they need internet access.

## Repository structure

```
analysis.py         full analysis as a script
analysis.ipynb      same analysis as a notebook, with outputs
download_data.py    downloads the dataset (for the notebook)
environment.yml     conda environment
```
