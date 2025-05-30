#%%
import torch
import esm
import argparse
from time import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

parser = argparse.ArgumentParser(description='Predict amyloidogenicity of fragments in a protein sequence.')
# Create mutually exclusive group for sequence, embeddingsFile, and embeddingsDir
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--sequence', '-s', help='Either a string of a peptide sequence (uppercase one-letter residue code) or a fasta file (.fasta or .fa) containing protein sequence(s). Use this if you do not already have embeddings for your sequence(s).')
group.add_argument('--embeddingsFiles', nargs='+', help='One or more pre-computed ESM embeddings files (.pt) with "mean_representations". Accepts glob patterns (e.g. "*.pt").')
parser.add_argument('--nogpu', action='store_true', help='Disable GPU usage. May be useful if you have memory issues.')
parser.add_argument('--output','-o', help='Output file name for predictions. Default is amyloidogenicity.csv', default='amyloidogenicity.csv')
# parser.add_argument('--ESM_model', help='Size of ESM model to use. Options: currently only "3B" model is supported.', default='3B')
parser.add_argument('--classifiers', nargs='+', choices=['6aa-FETA','6aa','10aa', '15aa', 'general'], default=['general'],
                    help="Classifier(s) to use. Options: '6aa-FETA','6aa', '10aa', '15aa', 'general'. Default is 'general'.")
parser.add_argument('--model_dir', help="Directory containing model files. Default is 'model_develpment/models'.", default='model_development/models')
parser.add_argument('--embeddings_output', help='Output file name for embeddings if you want them. Will save as .npy')
parser.add_argument('--no_plot', action='store_true', help='Disable plotting of predictions.')
args = parser.parse_args()

# Example usage:
# `python predict.py -s VQIVYK` will extract embeddings and output predictions for the single hexapeptide sequence
# `python predict.py -s example.fasta` will extract embeddings and output predictions for each sequence in the fasta file
# `python predict.py --embeddingsFiles example_embeddings_dir/example_IDR_1.pt --classifiers 6aa` will output predictions (using the 6aa-trained model) for the pre-computed embeddings in the .pt file
# `python predict.py --embeddingsFiles example_embeddings_dir/*.pt --classifiers general` will output predictions (using the general model) for all the pre-computed embeddings in the dir

# Process classifiers argument
classifiers_to_use = args.classifiers

# Load classification models as per user's choice
current_directory = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.realpath(args.model_dir)
models = {}

if '6aa-FETA' in classifiers_to_use:
    models['6aa-FETA'] = joblib.load(f'{model_dir}/6aa_FETA_model_latest.joblib')
if '6aa' in classifiers_to_use:
    models['6aa'] = joblib.load(f'{model_dir}/6aa_best_model_latest.joblib')
if '10aa' in classifiers_to_use:
    models['10aa'] = joblib.load(f'{model_dir}/10aa_model_latest.joblib')
if '15aa' in classifiers_to_use:
    models['15aa'] = joblib.load(f'{model_dir}/15aa_model_latest.joblib')
if 'general' in classifiers_to_use:
    models['general'] = joblib.load(f'{model_dir}/general_model_latest.joblib')

#%%

layer = 36 # number of layers in the ESM2 3B model's transformer
if args.sequence:
    tik = time()
    print('Extracting embeddings...')
    if args.sequence.endswith('.fasta') or args.sequence.endswith('.fa'):
        print('Processing fasta file...')
        # get a list of names and sequences
        with open(args.sequence) as f:
            lines = f.readlines()
        names = []
        sequences = []
        for line in lines:
            if line.startswith('>'):
                names.append(line[1:].strip())
            else:
                sequences.append(line.strip())
        if len(names) != len(sequences):
            raise ValueError('Number of names and sequences do not match in fasta file.')
    else:
        print('Processing single sequence...')
        sequences = [args.sequence]
        names = [args.sequence]
    
    # Load ESM2 model (3B parameter model)
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    
    # turn names and sequences into data
    data = [(name, sequence) for name, sequence in zip(names, sequences)]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
        batch_tokens = batch_tokens.cuda()  # Transfer the input tensor to GPU
    
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    tok = time()
    print(f'Time to load ESM model and preprocess data: {tok-tik:.0f} s')
    
    tik = time()
    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
    token_representations = results["representations"][layer]
    tok = time()
    print(f'Time to extract embeddings: {tok-tik:.2f} s')
    
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    sequence_representations = torch.stack(sequence_representations).cpu().numpy()
    
    # save embeddings 
    if args.embeddings_output:
        np.save(args.embeddings_output, sequence_representations)
        print(f'Embeddings saved to {args.embeddings_output}')

elif args.embeddingsFiles:
    print('Processing embeddings files...')
    # Expand glob patterns and collect all files
    all_files = []
    for pattern in args.embeddingsFiles:
        matched_files = glob.glob(pattern)
        if not matched_files:
            print(f"Warning: No files matched the pattern {pattern}")
        all_files.extend(matched_files)

    if not all_files:
        raise ValueError("No embeddings files found (expecting .pt files). Please check your patterns.")

    embeddings_list = []
    names = []
    for filepath in all_files:
        if filepath.endswith('.pt'):
            data = torch.load(filepath, weights_only=True)
            if 'mean_representations' in data:
                if layer in data['mean_representations']:
                    embeddings = data['mean_representations'][layer].numpy()
                    embeddings_list.append(embeddings)
                    name = os.path.splitext(os.path.basename(filepath))[0]
                    names.append(name)
                else:
                    raise ValueError(f"The 'mean_representations' in {filepath} does not contain layer {layer}.")
            else:
                raise ValueError(f"File {filepath} does not contain 'mean_representations'. Use the --include mean option when extracting embeddings with extract.py")
        else:
            print(f"Warning: {filepath} is not a .pt file. Skipping.")
    if not embeddings_list:
        raise ValueError('No valid .pt embeddings files found.')
    sequence_representations = np.stack(embeddings_list, axis=0)

# put into df
columns = [f'embedding_{i}' for i in range(sequence_representations.shape[1])]
embeddings_df = pd.DataFrame(sequence_representations, index=names, columns=columns)
X = embeddings_df

# Initialize dictionaries to store scores and logits
scores = {}
logits = {}

print("\nGenerating predictions for selected classifiers...")
for classifier_name in classifiers_to_use:
    if classifier_name in models.keys():
        print(f"Generating predictions using {classifier_name} model...")
        
        # Load model and associated data
        loaded_data = models[classifier_name]
        LR_model = loaded_data['model']
        scaler = loaded_data['scaler']
        
        if classifier_name == 'general':
            embeddings_scaled = scaler.transform(embeddings_df.values)
        else:
            embeddings_scaled = scaler.transform(embeddings_df)

        # Handle feature selection (for 6aa-FETA and 6aa-best)
        if classifier_name in ['6aa-FETA', '6aa']:
            if 'feature_set' not in loaded_data:
                raise ValueError(f"'feature_set' is missing in the {classifier_name} model file.")
            feature_set = loaded_data['feature_set']
            if isinstance(feature_set, tuple):
                feature_set = list(feature_set)
            embeddings_scaled = embeddings_scaled[:, feature_set]  # Subset to selected features

        # Make predictions
        predictions = LR_model.predict_proba(embeddings_scaled)[:, 1]
        scores[classifier_name] = np.round(predictions, 3)
    else:
        raise ValueError(f"Classifier '{classifier_name}' not found in the loaded models. Please check your model directory.")

# Print results
for i, name in enumerate(names):
    print(f'\n{name}')
    for classifier in classifiers_to_use:
        score = scores[classifier][i]
        print(f'{classifier} model score: {score:.3f}')

# Prepare output DataFrame
output_df = pd.DataFrame({'name': names})
for classifier in classifiers_to_use:
    output_df[f'{classifier}_score'] = scores[classifier]
output_df.to_csv(args.output, index=False)
print('Predictions saved to', args.output)

# Plotting predictions if there are multiple sequences
if len(names) > 1 and not args.no_plot:
    print("\nGenerating bar graphs for predictions...")
    for classifier in classifiers_to_use:
        if classifier in scores:
            plt.figure(figsize=(10, 6))
            plt.bar(names, scores[classifier], color='skyblue', alpha=0.8)
            plt.title(f"{classifier} Model Predictions", fontsize=16)
            plt.xlabel("Sequence Name", fontsize=14)
            plt.ylabel("Amyloidogenicity Score", fontsize=14)
            plt.xticks(rotation=45, fontsize=12, ha='right')
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()
else:
    print("\nSingle sequence detected. Skipping bar graphs.")
