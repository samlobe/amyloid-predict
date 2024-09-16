#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def precompute_go_term_to_idr_mapping(df, category):
    """
    Precomputes a mapping of GO terms to their associated IDRs.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing GO terms for each IDR.
    - category (str): The GO category, either 'molfunc' or 'cellcom'.
    
    Returns:
    - go_term_to_idrs (dict): Dictionary mapping GO terms to a list of IDRs.
    """
    column = 'GO_molfunc' if category == 'molfunc' else 'GO_cellcom' if category == 'cellcom' else None
    if column is None:
        raise ValueError(f"Invalid category: {category}. Choose 'molfunc' or 'cellcom'.")

    # Reset the index so that 'IDR' becomes a regular column
    df = df.reset_index()

    # Explode the GO term lists so each term is a separate row
    exploded_df = df[['IDR', column]].explode(column)

    # Create a dictionary mapping GO terms to the list of associated IDRs
    go_term_to_idrs = exploded_df.groupby(column)['IDR'].apply(list).to_dict()

    return go_term_to_idrs

def plot_histograms_for_go_terms_precomputed(go_term_to_idrs, scores_df, go_terms, category, bins=np.linspace(0, 1, 101)):
    """
    Plots histograms for all the GO terms in the specified category using precomputed IDR mappings.
    
    Args:
    - go_term_to_idrs (dict): Precomputed dictionary mapping GO terms to IDRs.
    - scores_df (pd.DataFrame): DataFrame containing amyloidogenicity scores for each IDR.
    - go_terms (list): List of GO terms to process.
    - category (str): The GO category, either 'molfunc' or 'cellcom'.
    - bins (np.array): Bins for the histogram.
    """
    for term in tqdm(go_terms):
        # Use the precomputed mapping to get the IDRs with the current GO term
        with_term = go_term_to_idrs.get(term, [])
        without_term = scores_df[~scores_df['IDR'].isin(with_term)]['IDR'].unique().tolist()

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
        plt.xlabel("Amyloidogenicity Score", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(False)
        plt.ylim(0, 3)
        plt.xlim(0, 1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Show the plot
        plt.show()

def save_histograms_for_go_terms(go_term_to_idrs, scores_df, go_terms, category, bins=np.linspace(0, 1, 101)):
    """
    Saves histograms (bin heights) for all the GO terms in the specified category using precomputed IDR mappings.
    Also adds the global histogram for all IDRs.
    
    Args:
    - go_term_to_idrs (dict): Precomputed dictionary mapping GO terms to IDRs.
    - scores_df (pd.DataFrame): DataFrame containing amyloidogenicity scores for each IDR.
    - go_terms (list): List of GO terms to process.
    - category (str): The GO category, either 'molfunc' or 'cellcom'.
    - bins (np.array): Bins for the histogram.
    
    Returns:
    - results (list): A list of dictionaries containing the bin heights for each GO term and global histogram.
    """
    results = []  # To store histogram data for each term

    # Calculate the global histogram for all IDRs
    all_idr_hist, _ = np.histogram(scores_df['mean_score'], bins=bins, density=True)
    results.append({
        'GO_term': 'ALL_IDRs',  # Special label for the global histogram
        'category': category,
        'bin_edges': bins[:-1],
        'with_term_hist': all_idr_hist,  # This is the global histogram
        'without_term_hist': np.zeros_like(all_idr_hist)  # No "without term" for global histogram
    })

    # Process each GO term
    for term in tqdm(go_terms):
        with_term = go_term_to_idrs.get(term, [])
        without_term = scores_df[~scores_df['IDR'].isin(with_term)]['IDR'].unique().tolist()

        # Filter scores_df based on the IDRs found
        with_term_df = scores_df[scores_df['IDR'].isin(with_term)]
        without_term_df = scores_df[scores_df['IDR'].isin(without_term)]

        # Check if there's enough data to calculate histograms
        if with_term_df.empty or without_term_df.empty:
            print(f"Skipping term {term} due to lack of data.")
            continue

        # Calculate the histograms using numpy.histogram
        with_term_hist, _ = np.histogram(with_term_df['mean_score'], bins=bins, density=True)
        without_term_hist, _ = np.histogram(without_term_df['mean_score'], bins=bins, density=True)

        # Store the results in a dictionary for each term
        result = {
            'GO_term': term,
            'category': category,
            'bin_edges': bins[:-1],  # Exclude the last bin edge since we use bin counts
            'with_term_hist': with_term_hist,
            'without_term_hist': without_term_hist
        }
        results.append(result)

    return results

# Save histograms to CSV
def save_histogram_data_to_csv(histogram_data, filename):
    """
    Saves the histogram data to a CSV file.
    
    Args:
    - histogram_data (list): List of dictionaries containing histogram data.
    - filename (str): The filename for the CSV output.
    """
    rows = []
    for data in histogram_data:
        for i, bin_edge in enumerate(data['bin_edges']):
            rows.append({
                'GO_term': data['GO_term'],
                'category': data['category'],
                'bin_edge': bin_edge,
                'with_term_hist': data['with_term_hist'][i],
                'without_term_hist': data['without_term_hist'][i]
            })

    # Convert to a DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved histogram data to {filename}")

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

# Precompute mappings for molecular function and cellular component GO terms
molfunc_to_idrs = precompute_go_term_to_idr_mapping(df_GO_combined, 'molfunc')
cellcom_to_idrs = precompute_go_term_to_idr_mapping(df_GO_combined, 'cellcom')

#%%
# Plot histograms for molecular function GO terms
print("Processing molecular function (molfunc) GO terms...")
plot_histograms_for_go_terms_precomputed(molfunc_to_idrs, scores_df, molfunc_terms, category='molfunc')

# Plot histograms for cellular component GO terms
print("Processing cellular component (cellcom) GO terms...")
plot_histograms_for_go_terms_precomputed(cellcom_to_idrs, scores_df, cellcom_terms, category='cellcom')

#%%
# Process and save histograms for molecular function GO terms
print("Processing molecular function (molfunc) GO terms...")
molfunc_histograms = save_histograms_for_go_terms(molfunc_to_idrs, scores_df, molfunc_terms, category='molfunc')
save_histogram_data_to_csv(molfunc_histograms, 'molfunc_histograms.csv')

# Process and save histograms for cellular component GO terms
print("Processing cellular component (cellcom) GO terms...")
cellcom_histograms = save_histograms_for_go_terms(cellcom_to_idrs, scores_df, cellcom_terms, category='cellcom')
save_histogram_data_to_csv(cellcom_histograms, 'cellcom_histograms.csv')
