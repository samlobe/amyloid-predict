#%%
import torch
import esm
import argparse
from time import time

parser = argparse.ArgumentParser(description='Predict amyloidogenicity of fragments in a protein sequence.')
parser.add_argument('sequence', help='Either a string of a peptide sequence (uppercase one-letter codes) or a fasta file containing protein sequence(s).')
parser.add_argument('--nogpu', action='store_true', help='Disable GPU usage.')

# debugging
args = parser.parse_args(['VQIVYK'])
args = parser.parse_args(["test.fasta"])

### load classification models
#
#
#
#

#%%
tik = time()
if args.sequence.endswith('.fasta') or ars.sequence.endswith('.fa'):
    # get a list of names and sequences
    with open(sequence) as f:
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

if torch.cuda.is_available() and not args.nogpu:
    model = model.cuda()
    print("Transferred model to GPU")

# turn names and sequences into data
data = [(name, sequence) for name, sequence in zip(names, sequences)]

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
tok = time()
print(f'Time to load model and preprocess data: {tok-tik:.2f} s')

#%%
tik = time()
# Extract per-residue representations
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[36], return_contacts=False)
token_representations = results["representations"][36]
tok = time()
print(f'Time to extract per-token embeddings: {tok-tik:.2f} s')

#%%
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

#%%
