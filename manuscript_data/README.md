# Manuscript Data
This directory contains manuscript-supporting datasets and scripts for the 2026 manuscript: "amyloid-predict and LLPS-predict: Predicting Phase Separation Propensities in the Intrinsically Disordered Proteome".

## Repository and Archive Policy
- Canonical repository: `amyloid-predict`
- Companion software repository: `LLPS-predict` 
- Data license for files in `manuscript_data/data/`: **CC BY 4.0**
- Code license: see top-level `LICENSE` in this repository
- DOI: **pending Zenodo minting at release `v1.0.0-manuscript`**
- Planned DOI landing page placeholder: `https://doi.org/TBD`

## Included in GitHub (small, reproducible)
- `data/snp/snp_features_damyloid.csv`
- `data/snp/updated_amyloid_scores.csv`
- `data/snp/or_tail_analysis_results.csv`
- `data/idrome/IDRome_IDRs.fasta` (IDR sequence FASTA used for fragment generation)
- `data/go/idr_go_categories_simple.csv` (simple GO categories per IDR)
- `data/go/amyloid_cellcom_histograms.csv`
- `data/go/LLPS_cellcom_histograms.csv`
- `data/go/amyloid_molfunc_histograms.csv`
- `data/go/LLPS_molfunc_histograms.csv`
- `data/go/cellcom_over_half.csv`
- `data/go/molfunc_over_half.csv`
- `data/residue_distributions/residue_code_score_distribution_summary.csv`
- `examples/joint_2d_plot/plot_joint_2d.py`
- `examples/joint_2d_plot/README.md`
- `examples/joint_2d_plot/hist.npy`
- `examples/joint_2d_plot/xedges.npy`
- `examples/joint_2d_plot/yedges.npy`
- `examples/go_fig3/plot_go_over_half_scatter.py`
- `examples/residue_boxenplots/reproduce_residue_boxenplots.py`

See Zenodo for other large raw tables.

## GO Analysis Reproducibility
- Included here:
  - Per-IDR GO category table (`data/go/idr_go_categories_simple.csv`)
  - Figure-3 GO histogram inputs (`data/go/*_histograms.csv`)
  - Figure-3 GO over-threshold summaries (`data/go/*_over_half.csv`)
  - A lightweight GO plotting helper (`examples/go_fig3/plot_go_over_half_scatter.py`)
- See Zenodo for the large residue-level score table used upstream for full GO derivations (`IDRs_amyloid_LLPS.csv`)

## Planned Zenodo Assets (Exact Filenames)

| Zenodo asset filename | Source file (local provenance) | SHA256 | Usage mapping |
|---|---|---|---|
| `fig4_IDRs_amyloid_LLPS_joint.csv` | `amyloid_LLPS_histogramming_Fig4/IDRs_amyloid_LLPS_joint.csv` | `812c2128a87f19ce7c85f61dd57264231c9f014017b18c2619f12f0bb9e42871` | Joint IDR table for Fig. 4 background/joint analyses |
| `fig4_IDRs_amyloid_LLPS.csv` | `amyloid_LLPS_histogramming_Fig4/IDRs_amyloid_LLPS.csv` | `78155f524b327abc2a06a80c49eef70bd61c5505fa1feb65d92d3a5047eb1c05` | Combined per-IDR score table used in Fig. 4 style analyses |
| `fig4_IDRs_amyloid_scores_general.csv` | `amyloid_LLPS_histogramming_Fig4/IDRs_amyloid_scores_general.csv` | `d3c435394892628aaac047289c99b907bf82da91bb74e3b31766530eb6adaed4` | Amyloid score component table |
| `fig4_IDRs_LLPS_scores.csv` | `amyloid_LLPS_histogramming_Fig4/IDRs_LLPS_scores.csv` | `57ba8f2a20a5f5d7e6cacb0a479fbca046db4a1fdfd4bca9c6d8e42bb9b5d764` | LLPS score component table |



### Score per-residue profiles with amyloid-predict and LLPS-predict (sample FASTA), then plot
```bash
# Run in amyloid-predict environment
amyloid-predict-per-res \
  --sequence example_single_sequence.fasta \
  --probe_lengths 6 10 15 \
  --stride 1 \
  --output sample_amyloid_perRes.csv

# Run in LLPS-predict environment
llps-predict-per-res \
  --sequence example_single_sequence.fasta \
  --probe_lengths 15 25 40 \
  --stride 1 \
  --output sample_LLPS_perRes.csv

python manuscript_data/examples/joint_2d_plot/plot_joint_2d.py \
  --amyloid_csv sample_amyloid_perRes.csv \
  --llps_csv sample_LLPS_perRes.csv \
  --protein sample \
  --out_png sample_joint_2d.png
```
