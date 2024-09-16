#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_idr_scores(idr_name, combined_per_res_scores, residue_range=None):
    """
    Plots 6aa, 10aa, and 15aa scores for a given IDR name, optionally within a specified residue range.
    
    Args:
    - idr_name (str): The name of the IDR to filter and plot. Example: "A0A024RBG1" or "A0A024RBG1_145_181".
    - residue_range (tuple): A tuple (start_res, end_res) specifying the residue range to plot. If None, the whole IDR is plotted.
    
    """
    # Split the IDR name if a range is provided
    if '_' in idr_name:
        base_idr, start_res, end_res = idr_name.split('_')
        start_res, end_res = int(start_res), int(end_res)
        residue_range = (start_res, end_res)
    else:
        base_idr = idr_name
    
    # Filter the DataFrame based on the IDR name
    filtered_df = combined_per_res_scores[combined_per_res_scores['IDR'].str.startswith(base_idr)]
    
    # If a residue range is provided, filter by that range
    if residue_range:
        filtered_df = filtered_df[(filtered_df['Residue'] >= residue_range[0]) & (filtered_df['Residue'] <= residue_range[1])]
    
    # Sort by residue number to ensure the x-axis is ordered
    filtered_df = filtered_df.sort_values('Residue')
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['Residue'], filtered_df['Score 6aa'], label='6aa Score', marker='o')
    plt.scatter(filtered_df['Residue'], filtered_df['Score 10aa'], label='10aa Score', marker='o')
    plt.scatter(filtered_df['Residue'], filtered_df['Score 15aa'], label='15aa Score', marker='o')

    # Add titles and labels
    plt.title(f"{idr_name}", fontsize=16)
    plt.xlabel('Residue Number', fontsize=14)
    plt.ylabel('Amyloidogenicity Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.show()

def find_idrs_by_go_term(df, term, category):
    """
    Finds IDRs with and without the GO term in the specified GO category.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing GO terms for each IDR.
    - term (str): The GO term to search for.
    - category (str): The GO category, either 'molfunc' or 'cellcom'.
    
    Returns:
    - with_term (list): List of IDRs with the GO term.
    - without_term (list): List of IDRs without the GO term.
    """
    column = 'GO_molfunc' if category == 'molfunc' else 'GO_cellcom' if category == 'cellcom' else None
    
    if column is None:
        raise ValueError(f"Invalid category: {category}. Choose 'molfunc' or 'cellcom'.")
    
    with_term = df[df[column].apply(lambda x: term in x if isinstance(x, list) else False)].index.tolist()
    without_term = df[df[column].apply(lambda x: term not in x if isinstance(x, list) else True)].index.tolist()

    return with_term, without_term

def plot_histograms_for_go_terms(df_GO_combined, scores_df, go_terms, category, bins=np.linspace(0, 1, 101)):
    """
    Plots histograms for all the GO terms in the specified category.
    
    Args:
    - df_GO_combined (pd.DataFrame): DataFrame containing GO terms for each IDR.
    - scores_df (pd.DataFrame): DataFrame containing amyloidogenicity scores for each IDR.
    - go_terms (list): List of GO terms to process.
    - category (str): The GO category, either 'molfunc' or 'cellcom'.
    - bins (np.array): Bins for the histogram.
    """
    for term in go_terms:
        # Find IDRs with and without the current GO term
        with_term, without_term = find_idrs_by_go_term(df_GO_combined, term, category)

        # Filter scores_df based on the IDRs found
        with_term_df = scores_df[scores_df['IDR'].isin(with_term)]
        without_term_df = scores_df[scores_df['IDR'].isin(without_term)]

        # Check if there's enough data to plot
        if with_term_df.empty or without_term_df.empty:
            print(f"Skipping term {term} due to lack of data.")
            continue

        # Plot histograms
        plt.figure(figsize=(10, 6))
        plt.hist(with_term_df['mean_score'], bins=bins, alpha=0.5, label=f'IDRs in {term} proteins', color='tab:blue', density=True)
        plt.hist(without_term_df['mean_score'], bins=bins, alpha=0.5, label=f'all other IDRs', color='tab:orange', density=True)

        # Add titles and labels
        # plt.title(f"Comparing {term} vs\n{term} proteins ({category})", fontsize=16)
        plt.xlabel("Amyloidogenicity Score", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(False)
        plt.ylim(0, 3)
        plt.xlim(0,1)
        plt.xticks(fontsize=14); plt.yticks(fontsize=14)

        # Show the plot
        plt.show()

# Load GO terms and sort them
df_GO_combined = pd.read_pickle('df_GO_combined.pkl')
molfunc_terms = df_GO_combined['GO_molfunc'].dropna().explode().unique().tolist()
cellcom_terms = df_GO_combined['GO_cellcom'].dropna().explode().unique().tolist()
molfunc_terms.sort()
cellcom_terms.sort()

# Load combined per-residue amyloidogenicity scores
scores_df = pd.read_pickle('combined_per_res_scores.pkl')

# Add column for max score in the 3 columns ("Score 6aa", "Score 10aa", "Score 15aa")
scores_df['max_score'] = scores_df[['Score 6aa', 'Score 10aa', 'Score 15aa']].max(axis=1)
# Add column for mean score in the 3 columns ("Score 6aa", "Score 10aa", "Score 15aa")
scores_df['mean_score'] = scores_df[['Score 6aa', 'Score 10aa', 'Score 15aa']].mean(axis=1)

# New block to find and plot IDRs for "G protein coupled receptor activity"
go_term = "G protein-coupled receptor activity"

# Find IDRs with the "G protein coupled receptor activity" GO term
gpcr_idrs, _ = find_idrs_by_go_term(df_GO_combined, go_term, category='molfunc')

#%%
for gpcr in gpcr_idrs[:5]:
    plot_idr_scores(gpcr,scores_df)

#%%
go_term = "growth factor activity"
gf_idrs = find_idrs_by_go_term(df_GO_combined, go_term, category='molfunc')
gf_high_score_counts = []; 
for gf in tqdm(gf_idrs[0]):
    # Filter the DataFrame to get the rows corresponding to the current GPCR
    filtered_gf_df = scores_df[scores_df['IDR'] == gf]
    count_above_80 = (filtered_gf_df['mean_score'] > 0.80).sum()
    gf_high_score_counts.append(count_above_80)

plt.hist(gf_high_score_counts)

#%%
which_high = [gf_idrs[0][i] for i, count in enumerate(gf_high_score_counts) if count > 15]
for gf in which_high:
    plot_idr_scores(gf,scores_df)

#%%
go_term = "receptor ligand activity"
rl_idrs = find_idrs_by_go_term(df_GO_combined, go_term, category='molfunc')

# %%
# Create a list to store the count of residues with a mean score above 0.80 for each GPCR
gpcr_high_score_counts = []

# Loop through each GPCR IDR
for gpcr in tqdm(gpcr_idrs):
    # Filter the DataFrame to get the rows corresponding to the current GPCR
    filtered_gpcr_df = scores_df[scores_df['IDR'] == gpcr]
    count_above_80 = (filtered_gpcr_df['mean_score'] > 0.80).sum()
    gpcr_high_score_counts.append(count_above_80)

#%%
# Plot the histogram of counts
plt.figure(figsize=(10, 6))
plt.hist(gpcr_high_score_counts)

# turn gpcr_high_score_counts into a pandas series
gpcr_high_score_counts = pd.Series(gpcr_high_score_counts)

#%%
# get a list of idrs with more than 20 high score counts
which_high = [gpcr_idrs[i] for i, count in enumerate(gpcr_high_score_counts) if count > 20]

# plot the high scoring IDRs by residue
for gpcr in which_high:
    plot_idr_scores(gpcr,scores_df)

#%%
print(which_high)