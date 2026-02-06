import argparse
import csv
import sys
from time import perf_counter

import torch

from amyloid_predict.inference import (
    CLASSIFIER_TO_CHECKPOINT,
    DEFAULT_MODEL_DIR,
    MODEL_NAME_TO_CHECKPOINT,
    MODEL_DIR_ENV_VAR,
    MODEL_NAME_TO_LAYER,
    configure_torch_hub_dir,
    describe_esm_checkpoint_state,
    embed_sequences,
    load_esm_model,
    load_inputs,
    load_lr_heads,
    predict_with_heads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict amyloidogenicity for one sequence or a FASTA file of sequences."
    )
    parser.add_argument(
        "--sequence",
        "-s",
        required=True,
        help="A protein sequence string, or a path to a FASTA file.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=sorted(CLASSIFIER_TO_CHECKPOINT.keys()),
        default=["general"],
        help="One or more classifier heads to run. Default: general",
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory containing LR .pt checkpoints. Relative paths are searched from both "
            f"the current working directory and the installed project root. Env override: {MODEL_DIR_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="amyloidogenicity.csv",
        help="Output CSV path. Default: amyloidogenicity.csv",
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    parser.add_argument(
        "--ESM_model",
        default="3B",
        choices=sorted(MODEL_NAME_TO_LAYER.keys()),
        help="ESM2 backbone to use. Currently supported: 3B",
    )
    parser.add_argument(
        "--esm_weights_dir",
        default=None,
        help=(
            "Optional custom Torch Hub directory for ESM weights cache. "
            "Weights are searched/downloaded at <dir>/checkpoints/."
        ),
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=4096,
        help=(
            "Maximum tokens per embedding batch. "
            "Higher values are faster but use more memory."
        ),
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="Truncate sequences longer than this length for ESM inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names, sequences = load_inputs(args.sequence)

    loading_t0 = perf_counter()

    hub_dir = configure_torch_hub_dir(args.esm_weights_dir)
    checkpoint_name = MODEL_NAME_TO_CHECKPOINT[args.ESM_model]
    checkpoint_path = describe_esm_checkpoint_state(hub_dir, checkpoint_name)

    esm_model, alphabet, layer = load_esm_model(args.ESM_model)
    heads = load_lr_heads(args.classifiers, args.model_dir)

    if checkpoint_path.exists():
        print(f"Using ESM2 checkpoint: {checkpoint_path}")

    loading_seconds = perf_counter() - loading_t0

    predict_t0 = perf_counter()

    use_gpu = torch.cuda.is_available() and not args.nogpu
    device_name = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device_name}")
    if not use_gpu:
        print("CPU mode detected. For faster runtime, see README GPU install instructions.")

    embeddings = embed_sequences(
        names=names,
        sequences=sequences,
        esm_model=esm_model,
        alphabet=alphabet,
        layer=layer,
        use_gpu=use_gpu,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
    )

    scores = predict_with_heads(embeddings, heads)

    prediction_seconds = perf_counter() - predict_t0

    if len(names) == 1:
        for classifier in args.classifiers:
            print(f"{classifier} score: {scores[classifier][0]:.4f}")

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        headers = ["Name"] + [f"{classifier}_score" for classifier in args.classifiers]
        writer.writerow(headers)
        for i, name in enumerate(names):
            row = [name] + [float(scores[classifier][i]) for classifier in args.classifiers]
            writer.writerow(row)

    print(f"Predictions saved to {args.output}")
    print(f"Weights loading time (s): {loading_seconds:.3f}")
    print(f"Prediction time (s): {prediction_seconds:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
