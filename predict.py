#%%
import torch
import esm
import argparse
from time import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Predict amyloidogenicity of fragments in a protein sequence.')
parser.add_argument('sequence', help='Either a string of a peptide sequence (uppercase one-letter residue code) or a fasta file (.fasta or .fa) containing protein sequence(s).')
parser.add_argument('--nogpu', action='store_true', help='Disable GPU usage.')
# args = parser.parse_args()

# debugging
args = parser.parse_args(['VQIVYK'])
# args = parser.parse_args(["example.fasta"])

### load classification models
model_dir = 'model_development/models_3B'
model_15aa = joblib.load(f'{model_dir}/tau_top_model.joblib') # trained on 15aa tau fragments from Louros et. al. 2024 (PAM4 paper)
model_6aa = joblib.load(f'{model_dir}/WALTZtht_top_model.joblib') # trained on 6aa peptides from WALTZdb, specifically those with Th-T data (http://waltzdb.switchlab.org)
model_10aa = joblib.load(f'{model_dir}/WALTZtht_top_model.joblib') # trained on 10aa fragments of PrP, lysozyme, and β-microglobulin from Fernandez-Escamilla, et. al. 2004 (TANGO paper)
ensemble_model = joblib.load(f'{model_dir}/ensemble_model.joblib') # uses logits from other 3 models for final prediction
selected_features = np.loadtxt(f'{model_dir}/selected_features_3B.csv', dtype=str)

#%%
tik = time()
if args.sequence.endswith('.fasta') or args.sequence.endswith('.fa'):
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
    sequences = [args.sequence]
    names = ['peptide1']

# Load ESM2 model
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
print(f'Time to load model and preprocess data: {tok-tik:.2f} s')

tik = time()
# Extract per-residue representations
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[36], return_contacts=False)
token_representations = results["representations"][36]
tok = time()
print(f'Time to extract per-token embeddings: {tok-tik:.2f} s')

sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

sequence_representations = torch.stack(sequence_representations).cpu().numpy()

# save embeddings
np.save('embeddings.npy', sequence_representations)

#%%
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
score_15aa = model_15aa.predict_proba(X)[:, 1] ; score_15aa = np.round(prob_15aa, 3)
score_6aa = model_6aa.predict_proba(X)[:, 1]; score_6aa = np.round(prob_6aa, 3)
score_10aa = model_10aa.predict_proba(X)[:, 1]; score_10aa = np.round(score_10aa, 3)

# print results
for name, pred in zip(names, ensemble_pred):
    print(f'{name}: {pred:.2f}')

# output csv with all 4 predictions
output_df = pd.DataFrame({'name': names, 'ensemble_score': ensemble_score, '15aa_model_score': score_15aa, '10aa_model_score': score_10aa, '6aa_model_score': score_6aa})
output_df.to_csv('amyloidogenicity.csv', index=False)

# plot bar graph of predictions
plt.figure()
plt.bar(names, ensemble_pred)
plt.ylabel('Amyloidogenicity')
plt.show()
