# amyloid-predict
Amyloidogenicity prediction with ESM2 embeddings and logistic-regression heads.

## Install
Choose one of the install paths below.

### Recommended (NVIDIA GPU, fastest)
```bash
conda create -n amyloid-predict python=3.9
conda activate amyloid-predict
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install fair-esm
pip install -e .
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"
```
Expected check output includes `cuda True`.

If this command fails, your driver/GPU/platform likely needs a different CUDA-enabled PyTorch build. Use the official selector to pick a compatible install command:
https://pytorch.org/get-started/locally/

### CPU fallback (Mac / no NVIDIA GPU, slower but supported)
```bash
conda create -n amyloid-predict python=3.9
conda activate amyloid-predict
pip install torch
pip install fair-esm
pip install -e .
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"
```
Expected check output includes `cuda False`.

## CLI Commands
After installation, two console commands are available:
- `amyloid-predict`
- `amyloid-predict-per-res`

## Usage
Single sequence score (default classifier: `general`):

```bash
amyloid-predict --sequence VQIVYK
```

Run multiple classifier heads at once:

```bash
amyloid-predict --sequence VQIVYK --classifiers general 6aa
```

FASTA with many sequences:

```bash
amyloid-predict --sequence example_multi_sequences.fasta --output example_sequences_amyloid_scores.csv
```

Tune embedding batching if needed (e.g. to avoid memory issues):

```bash
amyloid-predict \
  --sequence example_multi_sequences.fasta \
  --toks_per_batch 4096 \
  --truncation_seq_length 1022 \
  --output amyloidogenicity.csv
```

Per-residue profile for one sequence:

```bash
amyloid-predict-per-res \
  --sequence example_single_sequence.fasta \
  --probe_lengths 6 10 15 \
  --stride 1 \
  --output example_per_res_scores.csv
```

You can alter which classifiers to use for which fragments with the `--classifier_policy` and `--general_classifier` (see help with `amyloid-predict-per-res -h`)


## Notes
- `--toks_per_batch`: higher is faster but uses more memory.
- `--truncation_seq_length`: sequences longer than this are truncated for ESM2 inference.
- `--model_dir`: if relative, it is searched from both your current directory and the installed project root. You can also set `AMYLOID_PREDICT_MODEL_DIR`.

## Acknowledgments
- ESM developers
- Datasets used for training: WALTZ, TANGO, and tau fragment sets
- Scott Shell and Joan-Emma Shea
