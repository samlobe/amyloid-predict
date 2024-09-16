#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# load the residue scores
scores_df = pd.read_pickle('combined_per_res_scores.pkl')

#%%
def parse_fasta(fasta_file):
    """
    Parses a FASTA file and returns a dictionary of the sequences.
    """
    sequences = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_id = line[1:].strip()
                sequences[seq_id] = ''
            else:
                sequences[seq_id] += line.strip()
    return sequences

IDRs = parse_fasta('IDRs.fasta')
#%%
# Precompute the residue letters for each sequence in IDRs
def precompute_residues(IDRs):
    """
    Creates a dictionary that maps each IDR to a list of one-letter residues.
    """
    precomputed_residues = {}
    for key, sequence in IDRs.items():
        # Get just the IDR identifier (e.g., 'A0A024RBG1_145_181')
        idr_id = key.split('|')[0]
        start_pos = int(idr_id.split('_')[-2])
        precomputed_residues[idr_id] = {i + start_pos: residue for i, residue in enumerate(sequence)}
    return precomputed_residues

precomputed_residues = precompute_residues(IDRs)

# Now use the precomputed residue letters in your function
def get_residue_from_precomputed(row, precomputed_residues):
    """
    Retrieves the one-letter residue from the precomputed dictionary based on the IDR and residue number.
    """
    idr_id = row['IDR'].split('|')[0]  # Extract the IDR identifier
    residue_num = row['Residue']        # Get the residue number
    # Look up the precomputed residue
    return precomputed_residues.get(idr_id, {}).get(residue_num, np.nan)

tqdm.pandas()

# Apply the more efficient lookup to the DataFrame
scores_df['identity'] = scores_df.progress_apply(lambda row: get_residue_from_precomputed(row, precomputed_residues), axis=1)

# %%
# resave this
scores_df.to_pickle('combined_per_res_scores.pkl')
