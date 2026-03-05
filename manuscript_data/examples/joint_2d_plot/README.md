# Joint 2D Plot Repro (Per-Residue Amyloid + LLPS)

This folder reproduces the joint 2D residue trace plot from per-residue predictions.

## Inputs
- Per-residue amyloid CSV from `amyloid-predict-per-res`
- Per-residue LLPS CSV from `llps-predict-per-res` (see github.com/samlobe/LLPS-predict)
- Optional background density files:
  - `hist.npy`
  - `xedges.npy`
  - `yedges.npy`

## 1) Run per-residue scoring (amyloid)
Use the `amyloid-predict` environment and repo checkout.

```bash
conda activate amyloid-predict
amyloid-predict-per-res \
  --sequence example_single_sequence.fasta \
  --probe_lengths 6 10 15 \
  --stride 1 \
  --toks_per_batch 4096 \
  --output sample_amyloid_perRes.csv
```

## 2) Run per-residue scoring (LLPS)
Use the `LLPS-predict` environment and repo checkout.

```bash
conda activate LLPS-predict
llps-predict-per-res \
  --sequence example_single_sequence.fasta \
  --probe_lengths 15 25 40 \
  --stride 1 \
  --toks_per_batch 4096 \
  --output sample_LLPS_perRes.csv
```

## 3) Plot joint 2D trace
Run from `amyloid-predict` checkout.

### Minimal plot (no background density)
```bash
python manuscript_data/examples/joint_2d_plot/plot_joint_2d.py \
  --amyloid_csv sample_amyloid_perRes.csv \
  --llps_csv sample_LLPS_perRes.csv \
  --protein sample 
```

Default output filename for this command is `sample_joint_2d.png` in the current working directory.

### Plot with optional IDR background contours
```bash
python manuscript_data/examples/joint_2d_plot/plot_joint_2d.py \
  --amyloid_csv sample_amyloid_perRes.csv \
  --llps_csv sample_LLPS_perRes.csv \
  --hist_npy manuscript_data/examples/joint_2d_plot/hist.npy \
  --xedges_npy manuscript_data/examples/joint_2d_plot/xedges.npy \
  --yedges_npy manuscript_data/examples/joint_2d_plot/yedges.npy \
  --protein sample \
  --out_png manuscript_data/examples/joint_2d_plot/output/sample_joint_2d_with_hist.png
```

## Notes on chunking and runtime
- `*-predict-per-res` computes fragment windows defined by `--probe_lengths` and `--stride`.
- Large inputs are embedded in batches; `--toks_per_batch` controls memory/speed tradeoff.
- If you hit OOM, lower `--toks_per_batch` (for example `2048` or `1024`).
