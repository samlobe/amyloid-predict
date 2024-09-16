#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# load the residue scores
scores_df = pd.read_pickle('combined_per_res_scores.pkl')
# make a mean score column
scores_df['mean_score'] = scores_df[['Score 6aa', 'Score 10aa', 'Score 15aa']].mean(axis=1)

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

# load all human proteins (includes ordered and disordered regions)
human_proteins = parse_fasta('UP000005640_9606.fasta')
IDRs = parse_fasta('IDRs.fasta')

#%%
# concatenate all the human_proteins sequences into a single string
all_human_proteins = ''.join(human_proteins.values())

# count the number of each amino acid in the human proteome
all_protein_aa_counts = {aa: all_human_proteins.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}

# divide by the total number of amino acids to get the frequency
len = sum(all_protein_aa_counts.values())
all_protein_aa_probs = {aa: count/len for aa, count in all_protein_aa_counts.items()}

# plot the histogram of amino acid counts
plt.bar(all_protein_aa_probs.keys(), all_protein_aa_probs.values())
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.ylim(0,0.12)
plt.title('Human Proteome', fontsize=16)

#%%
all_IDRs = ''.join(IDRs.values())
all_IDR_aa_counts = {aa: all_IDRs.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
len = sum(all_IDR_aa_counts.values())
all_IDR_aa_probs = {aa: count/len for aa, count in all_IDR_aa_counts.items()}
plt.bar(all_IDR_aa_probs.keys(), all_IDR_aa_probs.values(),color='tab:orange')
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.ylim(0,0.12)
plt.title('Intrinsically Disordered Regions', fontsize=16)

#%%
# subtract the IDR amino acid frequencies from the proteome amino acid frequencies
aa_diff = {aa: all_protein_aa_probs[aa] - all_IDR_aa_probs[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
# Separate positive and negative values for coloring
colors = ['tab:blue' if v > 0 else 'tab:orange' for v in aa_diff.values()]
plt.bar(aa_diff.keys(), aa_diff.values(), color=colors)
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency Difference', fontsize=14)
plt.title('Proteome - IDRs', fontsize=16)
plt.axhline(0, color='black', linewidth=1)
plt.ylim(-0.03,0.03)

#%%
# look at amino acid composition of LLPS drivers
members = pd.read_csv('LLPS/members.csv', header=None)[0].to_list()
drivers = pd.read_csv('LLPS/drivers.csv', header=None)[0].to_list()

LLPS_drivers_scores = scores_df[scores_df['IDR'].str.startswith(tuple(drivers))] # 5 seconds
LLPS_members_scores = scores_df[scores_df['IDR'].str.startswith(tuple(members))] # 154 seconds

#%%
# histogram of amino acid frequencies in LLPS drivers
LLPS_drivers = ''.join(LLPS_drivers_scores['identity'].dropna())
LLPS_drivers_aa_counts = {aa: LLPS_drivers.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
len = sum(LLPS_drivers_aa_counts.values())
LLPS_drivers_aa_probs = {aa: count/len for aa, count in LLPS_drivers_aa_counts.items()}
plt.bar(LLPS_drivers_aa_probs.keys(), LLPS_drivers_aa_probs.values())
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
# plt.ylim(0,0.12)
plt.title('LLPS Drivers\n(from CD-CODE database)', fontsize=16)

#%%
# histogram of amino acid frequencies in LLPS members
LLPS_members = ''.join(LLPS_members_scores['identity'].dropna())
LLPS_members_aa_counts = {aa: LLPS_members.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
len = sum(LLPS_members_aa_counts.values())
LLPS_members_aa_probs = {aa: count/len for aa, count in LLPS_members_aa_counts.items()}
plt.bar(LLPS_members_aa_probs.keys(), LLPS_members_aa_probs.values())
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
# plt.ylim(0,0.12)
plt.title('LLPS Members\n(from CD-CODE database)', fontsize=16)

#%% plot the difference between LLPS drivers and all IDRs
aa_diff_drivers = {aa: LLPS_drivers_aa_probs[aa] - all_IDR_aa_probs[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
colors = ['tab:purple' if v > 0 else 'tab:orange' for v in aa_diff_drivers.values()]
plt.bar(aa_diff_drivers.keys(), aa_diff_drivers.values(),color=colors)
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency Difference', fontsize=14)
plt.title('LLPS Drivers - IDRs', fontsize=16)
plt.ylim(-0.03,0.03)
plt.axhline(0, color='black', linewidth=1)

#%% plot the difference between LLPS members and all IDRs
aa_diff_members = {aa: LLPS_members_aa_probs[aa] - all_IDR_aa_probs[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
colors = ['tab:pink' if v > 0 else 'tab:orange' for v in aa_diff_members.values()]
plt.bar(aa_diff_members.keys(), aa_diff_members.values(),color=colors)
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency Difference', fontsize=14)
plt.title('LLPS Members - IDRs', fontsize=16)
plt.ylim(-0.03,0.03)
plt.axhline(0, color='black', linewidth=1)

#%%
# Get the 5% of rows with the lowest mean amyloidogenicity scores
lowest_5_percent_threshold = scores_df['mean_score'].quantile(0.05)
lowest_5_percent_df = scores_df[scores_df['mean_score'] <= lowest_5_percent_threshold]

# Now we calculate the frequencies of the residues in the "identity" column
lowest_5_percent_residues = lowest_5_percent_df['identity'].value_counts()

# Convert the counts to frequencies
total_residues_lowest_5_percent = lowest_5_percent_residues.sum()
lowest_5_percent_aa_probs = {aa: count / total_residues_lowest_5_percent for aa, count in lowest_5_percent_residues.items()}

# Plot the amino acid frequencies for the lowest 5% amyloidogenicity residues
plt.bar(lowest_5_percent_aa_probs.keys(), lowest_5_percent_aa_probs.values(), color='tab:green')
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.ylim(0, 0.12)
plt.title('Amino Acid Frequencies in Lowest 5% Amyloidogenicity Scores', fontsize=16)
plt.show()

#%%
# get the difference between the lowest 5% and all IDRs
aa_diff_lowest_5_percent = {aa: lowest_5_percent_aa_probs[aa] - all_IDR_aa_probs[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
colors = ['tab:red' if v > 0 else 'tab:orange' for v in aa_diff_lowest_5_percent.values()]
plt.bar(aa_diff_lowest_5_percent.keys(), aa_diff_lowest_5_percent.values(),color=colors)
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency Difference', fontsize=14)
plt.title('Lowest 5% Amyloidogenicity Scores - all IDRs', fontsize=16)
plt.ylim(-0.08,0.14)
# do y-ticks by 0.01 increments
plt.yticks(np.arange(-0.08,0.14,0.02))
plt.axhline(0, color='black', linewidth=1)

#%%
# get the 5% of rows with the highest mean amyloidogenicity scores
highest_5_percent_threshold = scores_df['mean_score'].quantile(0.95)
highest_5_percent_df = scores_df[scores_df['mean_score'] >= highest_5_percent_threshold]

# Now we calculate the frequencies of the residues in the "identity" column
highest_5_percent_residues = highest_5_percent_df['identity'].value_counts()

# Convert the counts to frequencies
total_residues_highest_5_percent = highest_5_percent_residues.sum()
highest_5_percent_aa_probs = {aa: count / total_residues_highest_5_percent for aa, count in highest_5_percent_residues.items()}
plt.bar(highest_5_percent_aa_probs.keys(), highest_5_percent_aa_probs.values(), color='tab:purple')
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.ylim(0, 0.12)
plt.title('Amino Acid Frequencies in Highest 5% Amyloidogenicity Scores', fontsize=16)
plt.show()

#%%
# get the difference between the highest 5% and all IDRs
aa_diff_highest_5_percent = {aa: highest_5_percent_aa_probs[aa] - all_IDR_aa_probs[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
colors = ['tab:green' if v > 0 else 'tab:orange' for v in aa_diff_highest_5_percent.values()]
plt.bar(aa_diff_highest_5_percent.keys(), aa_diff_highest_5_percent.values(),color=colors)
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Frequency Difference', fontsize=14)
plt.title('Highest 5% Amyloidogenicity Scores - all IDRs', fontsize=16)
plt.ylim(-0.08,0.14)
# do y-ticks by 0.01 increments
plt.yticks(np.arange(-0.08,0.14,0.02))
plt.axhline(0, color='black', linewidth=1)

#%%
#%%
# Calculate the mean mean_score for each amino acid (A, C, D, E, etc.)
mean_score_by_residue = scores_df.groupby('identity')['mean_score'].mean()

# Convert to dictionary (optional, for easy lookup)
mean_score_by_residue_dict = mean_score_by_residue.to_dict()

# Print the mean score for each amino acid
for aa, mean_score in mean_score_by_residue_dict.items():
    print(f"Amino Acid: {aa}, Mean mean_score: {mean_score}")

# Plot the mean mean_score for each amino acid
plt.bar(mean_score_by_residue.index, mean_score_by_residue.values, color='tab:cyan')
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Mean Amyloidogenicity Score', fontsize=14)
plt.title('Mean Amyloidogenicity Score by Amino Acid', fontsize=16)
plt.show()

#%%
# sort from highest to lowest mean score and replot
mean_score_by_residue_sorted = mean_score_by_residue.sort_values(ascending=False)
plt.bar(mean_score_by_residue_sorted.index, mean_score_by_residue_sorted.values, color='tab:cyan')
plt.xlabel('Amino Acid', fontsize=14)
plt.ylabel('Mean Amyloidogenicity Score', fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Mean Amyloidogenicity Score by Amino Acid', fontsize=16)
plt.show()

#%%
# save a csv of all the compositions and scores I've collected so far:
# all_protein_aa_probs, all_IDR_aa_probs, LLPS_drivers_aa_probs, LLPS_members_aa_probs, lowest_5_percent_aa_probs, highest_5_percent_aa_probs, mean_score_by_residue
all_compositions = pd.DataFrame([all_protein_aa_probs, all_IDR_aa_probs, LLPS_drivers_aa_probs, LLPS_members_aa_probs, lowest_5_percent_aa_probs, highest_5_percent_aa_probs, mean_score_by_residue]).T
all_compositions.columns = ['all_protein', 'all_IDRs', 'LLPS_drivers', 'LLPS_members', 'lowest_5_percent', 'highest_5_percent', 'mean_score']
all_compositions.to_csv('compositions.csv')

#%%
def map_idrs_to_go_terms(df_GO_combined, category):
    """
    Maps each IDR to its associated GO terms.
    
    Args:
    - df_GO_combined (pd.DataFrame): The DataFrame containing GO terms for each IDR.
    - category (str): The GO category, either 'molfunc' or 'cellcom'.
    
    Returns:
    - idr_to_go_terms (dict): Dictionary mapping IDRs to their GO terms.
    """
    column = 'GO_molfunc' if category == 'molfunc' else 'GO_cellcom' if category == 'cellcom' else None
    if column is None:
        raise ValueError(f"Invalid category: {category}. Choose 'molfunc' or 'cellcom'.")
    
    df = df_GO_combined.reset_index()
    exploded_df = df[['IDR', column]].explode(column)
    idr_to_go_terms = exploded_df.groupby('IDR')[column].apply(set).to_dict()
    
    return idr_to_go_terms

df_GO_combined = pd.read_pickle('df_GO_combined.pkl')
molfunc_terms = df_GO_combined['GO_molfunc'].dropna().explode().unique().tolist()
cellcom_terms = df_GO_combined['GO_cellcom'].dropna().explode().unique().tolist()
molfunc_terms.sort()
cellcom_terms.sort()

# Precompute IDR to GO term mappings
idr_to_molfunc_terms = map_idrs_to_go_terms(df_GO_combined, 'molfunc')
idr_to_cellcom_terms = map_idrs_to_go_terms(df_GO_combined, 'cellcom')

# Create new columns in scores_df for molfunc and cellcom GO terms
scores_df['GO_molfunc'] = scores_df['IDR'].map(idr_to_molfunc_terms)
scores_df['GO_cellcom'] = scores_df['IDR'].map(idr_to_cellcom_terms)

# Fill NaN with empty sets to avoid issues with missing data
scores_df['GO_molfunc'] = scores_df['GO_molfunc'].apply(lambda x: x if isinstance(x, set) else set())
scores_df['GO_cellcom'] = scores_df['GO_cellcom'].apply(lambda x: x if isinstance(x, set) else set())

# Optimized function to calculate amino acid composition for GO terms
def calculate_aa_composition_by_go_term(scores_df, term, category):
    """
    Calculate amino acid composition for a GO term by filtering the precomputed GO terms.
    
    Args:
    - scores_df (pd.DataFrame): DataFrame containing the scores and precomputed GO terms.
    - term (str): The GO term to calculate amino acid composition for.
    - category (str): The GO category ('molfunc' or 'cellcom').
    
    Returns:
    - aa_probs (dict): Amino acid composition probabilities.
    """
    go_column = 'GO_molfunc' if category == 'molfunc' else 'GO_cellcom'
    
    # Filter rows where the GO term is present in the precomputed GO terms
    filtered_scores = scores_df[scores_df[go_column].apply(lambda go_terms: term in go_terms)]
    
    # Calculate amino acid composition for the filtered scores
    group_res = ''.join(filtered_scores['identity'].dropna().astype(str))
    aa_counts = {aa: group_res.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    total_len = sum(aa_counts.values())
    aa_probs = {aa: count / total_len if total_len > 0 else 0 for aa, count in aa_counts.items()}
    
    return aa_probs

# Process molecular function GO terms
print("Processing molecular function (molfunc) GO terms...")
molfunc_compositions = []
for term in tqdm(molfunc_terms):
    aa_probs = calculate_aa_composition_by_go_term(scores_df, term, category='molfunc')
    aa_probs['GO_term'] = term
    molfunc_compositions.append(aa_probs)

# Process cellular component GO terms
print("Processing cellular component (cellcom) GO terms...")
cellcom_compositions = []
for term in tqdm(cellcom_terms):
    aa_probs = calculate_aa_composition_by_go_term(scores_df, term, category='cellcom')
    aa_probs['GO_term'] = term
    cellcom_compositions.append(aa_probs)

#%%
# Save the results
molfunc_compositions_df = pd.DataFrame(molfunc_compositions)
cellcom_compositions_df = pd.DataFrame(cellcom_compositions)

# set the GO term as the index
molfunc_compositions_df.set_index('GO_term', inplace=True)
cellcom_compositions_df.set_index('GO_term', inplace=True)

molfunc_compositions_df.to_csv('molfunc_compositions.csv')
cellcom_compositions_df.to_csv('cellcom_compositions.csv')

#%%
from scipy.stats import entropy

def calculate_kl_divergence(aa_probs, full_idr_probs):
    """
    Calculates the Kullback-Leibler (KL) divergence between amino acid compositions of a GO category
    and the full IDR amino acid composition.
    
    Args:
    - aa_probs (dict): Amino acid probabilities for the GO term.
    - full_idr_probs (dict): Full IDR amino acid probabilities.
    
    Returns:
    - kl_divergence (float): KL divergence between the two compositions.
    """
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_probs_list = [aa_probs[aa] for aa in aa_list]
    full_idr_probs_list = [full_idr_probs[aa] for aa in aa_list]
    
    # Ensure there are no zero values to avoid log(0) issues
    aa_probs_list = [max(p, 1e-10) for p in aa_probs_list]
    full_idr_probs_list = [max(p, 1e-10) for p in full_idr_probs_list]
    
    return entropy(aa_probs_list, full_idr_probs_list)

for aa_probs in molfunc_compositions:
    term = aa_probs['GO_term']
    kl_div = calculate_kl_divergence(aa_probs, all_IDR_aa_probs)
    aa_probs['kl_divergence'] = kl_div

for aa_probs in cellcom_compositions:
    term = aa_probs['GO_term']
    kl_div = calculate_kl_divergence(aa_probs, all_IDR_aa_probs)
    aa_probs['kl_divergence'] = kl_div

