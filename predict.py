#%%
import torch
import esm
import argparse
from time import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Predict amyloidogenicity of fragments in a protein sequence.')
parser.add_argument('--sequence','-s', help='Either a string of a peptide sequence (uppercase one-letter residue code) or a fasta file (.fasta or .fa) containing protein sequence(s). Use this if you do not already have embeddings for your sequence(s).')
parser.add_argument('--embeddingsDir', help='Directory containing ESM embeddings files (.pt) with "mean_representations" for your sequence(s). Use this if you already have embeddings for your sequences (extract.py is a good way to get embeddings).')
parser.add_argument('--embeddingsFile', help='Pytorch file (.pt) containing ESM embeddings with "mean_representations" for a sequence. Use this if you already have ESM embeddings for your sequence (extract.py is a good way to get embeddings).', default=None)
parser.add_argument('--nogpu', action='store_true', help='Disable GPU usage. May be useful if you have memory issues.')
parser.add_argument('--output','-o', help='Output file name for predictions. Default is amyloidogenicity.csv', default='amyloidogenicity.csv')
parser.add_argument('--ESM_model', help='Size of ESM model to use. Options: "3B" or "15B". Default is 3B.', default='3B')
parser.add_argument('--classifiers', nargs='+', choices=['6aa', '10aa', '15aa', 'ensembled', 'all'], default=['all'],
                    help="Classifier(s) to use. Options: '6aa', '10aa', '15aa', 'ensembled', or 'all'.")
parser.add_argument('--embeddings_output', help='Output file name for embeddings if you want them. Will save as .npy')
args = parser.parse_args()

# Example usage:
# python predict.py -s VQIVYK
# python predict.py -s example.fasta
# python predict.py --embeddingsFile example_embeddings_dir/example_IDR_1.pt
# python predict.py --embeddingsDir example_embeddings_dir

# make sure args.model is valid
if args.ESM_model not in ['3B', '15B']:
    raise ValueError('Invalid model argument. Must be 3B or 15B.')
print(f'Using ESM {args.ESM_model} model.')

# Process classifiers argument
if 'all' in args.classifiers:
    classifiers_to_use = ['6aa', '10aa', '15aa', 'ensembled']
else:
    classifiers_to_use = args.classifiers

# Determine which models need to be loaded
models_to_load = set()
if '6aa' in classifiers_to_use or 'ensembled' in classifiers_to_use:
    models_to_load.add('6aa')
if '10aa' in classifiers_to_use or 'ensembled' in classifiers_to_use:
    models_to_load.add('10aa')
if '15aa' in classifiers_to_use or 'ensembled' in classifiers_to_use:
    models_to_load.add('15aa')
if 'ensembled' in classifiers_to_use:
    models_to_load.add('ensembled')

# Load classification models as per user's choice
current_directory = os.path.dirname(os.path.abspath(__file__))
model_dir= os.path.join(current_directory, "model_development", f"models_{args.ESM_model}")
model_dir = f'model_development/models_{args.ESM_model}'
models = {}
selected_features = np.loadtxt(f'{model_dir}/selected_features_{args.ESM_model}.csv', dtype=str)

if '6aa' in models_to_load:
    models['6aa'] = joblib.load(f'{model_dir}/WALTZtht_top_model.joblib')
if '10aa' in models_to_load:
    models['10aa'] = joblib.load(f'{model_dir}/TANGO_Table2_top_model.joblib')
if '15aa' in models_to_load:
    models['15aa'] = joblib.load(f'{model_dir}/tau_top_model.joblib')
if 'ensembled' in models_to_load:
    models['ensembled'] = joblib.load(f'{model_dir}/ensemble_model.joblib')

#%%
if args.sequence:
    tik = time()
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
    
    # Load ESM2 model
    if args.ESM_model == '3B':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        layer = 36
    else:
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        layer = 48
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
    print(f'Time to load ESM model and preprocess data: {tok-tik:.2f} s')
    
    tik = time()
    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
    token_representations = results["representations"][layer]
    tok = time()
    print(f'Time to extract per-token embeddings: {tok-tik:.2f} s')
    
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    sequence_representations = torch.stack(sequence_representations).cpu().numpy()
    
    # save embeddings 
    if args.embeddings_output:
        np.save(args.embeddings_output, sequence_representations)
        print(f'Embeddings saved to {args.embeddings_output}')

elif args.embeddingsFile:
    print('Processing embeddings file...')
    # process embeddingsFile
    if args.ESM_model == '3B':
        layer = 36
    else:
        layer = 48

    data = torch.load(args.embeddingsFile, weights_only=True)
    if 'mean_representations' in data:
        if layer in data['mean_representations']:
            embeddings = data['mean_representations'][layer].numpy()
            sequence_representations = embeddings[np.newaxis, :]  # ensure it's 2D array
            # set names appropriately
            basename = os.path.basename(args.embeddingsFile)
            name = os.path.splitext(basename)[0]
            names = [name]
        else:
            raise ValueError(f"The 'mean_representations' in {args.embeddingsFile} does not contain layer {layer}.")
    else:
        raise ValueError(f"File {args.embeddingsFile} does not contain 'mean_representations'. Please provide a .pt file with 'mean_representations'. You can use extract.py with the `--include mean` to get this.")


elif args.embeddingsDir:
    print('Processing embeddings directory...')
    # process embeddingsDir
    embeddings_list = []
    names = []
    if args.ESM_model == '3B':
        layer = 36
    else:
        layer = 48
    found_pt_files = False
    for filename in os.listdir(args.embeddingsDir):
        if filename.endswith('.pt'):
            found_pt_files = True
            filepath = os.path.join(args.embeddingsDir, filename)
            data = torch.load(filepath, weights_only=True)
            if 'mean_representations' in data:
                if layer in data['mean_representations']:
                    embeddings = data['mean_representations'][layer].numpy()
                    embeddings_list.append(embeddings)
                    name = os.path.splitext(filename)[0]
                    names.append(name)
                else:
                    raise ValueError(f"The 'mean_representations' in {filepath} does not contain layer {layer}.")
            else:
                raise ValueError(f"File {filepath} does not contain 'mean_representations'. Please provide .pt files with 'mean_representations'. You can use extract.py with the `--include mean` to get this.")
    if not found_pt_files:
        raise ValueError('No .pt files found in the specified embeddings directory.')
    sequence_representations = np.stack(embeddings_list, axis=0)

# put into df
columns = [f'embedding_{i}' for i in range(sequence_representations.shape[1])]
embeddings_df = pd.DataFrame(sequence_representations, index=names, columns=columns)
X = embeddings_df[selected_features]

# Initialize dictionaries to store scores and logits
scores = {}
logits = {}

# Compute predictions for each classifier
for model_name in ['6aa', '10aa', '15aa']:
    if model_name in models_to_load:
        model = models[model_name]
        logits_model = model.decision_function(X)
        logits[model_name] = logits_model
        score_model = model.predict_proba(X)[:, 1]
        scores[model_name] = np.round(score_model, 3)

# Compute ensemble model predictions if selected
if 'ensembled' in classifiers_to_use:
    # Ensure logits for individual models are available
    required_models = ['6aa', '10aa', '15aa']
    for req_model in required_models:
        if req_model not in logits:
            raise ValueError(f"Logits for model '{req_model}' are required for ensemble prediction but were not computed.")
    # Stack logits and compute ensemble predictions
    logits_array = np.vstack([logits['15aa'], logits['6aa'], logits['10aa']]).T
    ensemble_model = models['ensembled']
    ensemble_score = ensemble_model.predict_proba(logits_array)[:, 1]
    scores['ensembled'] = np.round(ensemble_score, 3)

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

# Plotting
if len(names) == 1:
    exit()

if 'ensembled' in classifiers_to_use:
    # Plot only the ensembled model
    plt.figure()
    plt.bar(names, scores['ensembled'], width=0.4, label='Ensembled model')
    plt.title(f'Ensembled model predictions\n(using ESM {args.ESM_model} model)')
    plt.ylabel('Amyloidogenicity')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    # Plot the selected classifiers ('6aa', '10aa', '15aa')
    plt.figure()
    x = np.arange(len(names))  # the label locations
    width = 0.2  # the width of the bars

    num_classifiers = len(classifiers_to_use)
    offsets = np.linspace(-width*num_classifiers/2, width*num_classifiers/2, num_classifiers, endpoint=False) + width/2

    for idx, classifier in enumerate(classifiers_to_use):
        plt.bar(x + offsets[idx], scores[classifier], width=width, label=f'{classifier} model')

    plt.xticks(x, names, rotation=90)
    plt.ylabel('Amyloidogenicity')
    plt.title(f'Model predictions\n(using ESM {args.ESM_model} model)')
    plt.legend()
    plt.tight_layout()
    plt.show()

