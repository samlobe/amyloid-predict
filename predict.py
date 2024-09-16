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
parser.add_argument('--nogpu', action='store_true', help='Disable GPU usage.')
parser.add_argument('--output','-o', help='Output file name for predictions. Default is amyloidogenicity.csv', default='amyloidogenicity.csv')
parser.add_argument('--embeddings_output', help='Output file name for embeddings. Default is embeddings.npy', default='embeddings.npy')
parser.add_argument('--model','-m', help='Size of ESM model to use. Options: "3B" or "15B". Default is 3B.', default='3B')
args = parser.parse_args()

# Example usage:
# python predict.py -s VQIVYK
# python predict.py -s example.fasta
# python predict.py --embeddingsFile "example_embeddings_dir/X6R8D5_1_127|X6R8D5.pt"
# python predict.py --embeddingsDir example_embeddings_dir

# make sure args.model is valid
if args.model not in ['3B', '15B']:
    raise ValueError('Invalid model argument. Must be 3B or 15B.')
print(f'Using ESM {args.model} model.')

# make sure only one of --sequence, --embeddingsFile, or --embeddingsDir is specified
if sum([bool(args.sequence), bool(args.embeddingsFile), bool(args.embeddingsDir)]) != 1:
    raise ValueError('Only one of --sequence, --embeddingsFile, or --embeddingsDir must be specified.')

### load classification models
model_dir = f'model_development/models_{args.model}'
model_15aa = joblib.load(f'{model_dir}/tau_top_model.joblib') # trained on 15aa tau fragments from Louros et. al. 2024 (PAM4 paper)
model_6aa = joblib.load(f'{model_dir}/WALTZtht_top_model.joblib') # trained on 6aa peptides from WALTZdb, specifically those with Th-T data (http://waltzdb.switchlab.org)
model_10aa = joblib.load(f'{model_dir}/TANGO_Table2_top_model.joblib') # trained on 10aa fragments of PrP, lysozyme, and β-microglobulin from Fernandez-Escamilla, et. al. 2004 (TANGO paper)
ensemble_model = joblib.load(f'{model_dir}/ensemble_model.joblib') # uses logits from other 3 models for final prediction
selected_features = np.loadtxt(f'{model_dir}/selected_features_{args.model}.csv', dtype=str)

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
    if args.model == '3B':
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
    print(f'Time to load model and preprocess data: {tok-tik:.2f} s')
    
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
    np.save(args.embeddings_output, sequence_representations)

elif args.embeddingsFile:
    print('Processing embeddings file...')
    # process embeddingsFile
    if args.model == '3B':
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
    if args.model == '3B':
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

else:
    raise ValueError('One of --sequence, --embeddingsFile, or --embeddingsDir must be specified.')


# put into df
columns = [f'embedding_{i}' for i in range(sequence_representations.shape[1])]
embeddings_df = pd.DataFrame(sequence_representations, index=names, columns=columns)
X = embeddings_df[selected_features]

logits_15aa = model_15aa.decision_function(X)
logits_6aa = model_6aa.decision_function(X)
logits_10aa = model_10aa.decision_function(X)
logits = np.vstack([logits_15aa, logits_6aa, logits_10aa]).T

ensemble_score = ensemble_model.predict_proba(logits)[:, 1] # probability of being amyloidogenic
ensemble_score = np.round(ensemble_score, 3)
score_15aa = model_15aa.predict_proba(X)[:, 1] ; score_15aa = np.round(score_15aa, 3)
score_6aa = model_6aa.predict_proba(X)[:, 1]; score_6aa = np.round(score_6aa, 3)
score_10aa = model_10aa.predict_proba(X)[:, 1]; score_10aa = np.round(score_10aa, 3)

# print results
for name, predE, pred6, pred10, pred15 in zip(names, ensemble_score, score_15aa, score_6aa, score_10aa):
    print(f'\n{name}')
    print(f'Ensemble score: {predE:.3f}')
    print(f'15aa model score: {pred15:.3f}')
    print(f'10aa model score: {pred10:.3f}')
    print(f'6aa model score: {pred6:.3f}')

# output csv with all 4 predictions
output_df = pd.DataFrame({'name': names, 'ensemble_score': ensemble_score, '15aa_model_score': score_15aa, '10aa_model_score': score_10aa, '6aa_model_score': score_6aa})
output_df.to_csv(args.output, index=False)
print('Predictions saved to ', args.output)

# if multiple sequences are provided, plot the ensemble model predictions
if len(names) == 1:
    exit()
# plot bar graph of predictions
plt.figure()
# bar width should be small
plt.bar(names, ensemble_score, width=0.4, label='Ensemble model')
plt.title(f'ensemble model predictions\n(using {args.model} model)')
plt.ylabel('amyloidogenicity')
# rotate the x-ticks 90 degrees
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
