#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined per-residue scores
combined_per_res_scores = pd.read_pickle('combined_per_res_scores.pkl')

def plot_idr_scores(idr_name, residue_range=None):
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
    plt.plot(filtered_df['Residue'], filtered_df['Score 6aa'], label='6aa Score', marker='o')
    plt.plot(filtered_df['Residue'], filtered_df['Score 10aa'], label='10aa Score', marker='o')
    plt.plot(filtered_df['Residue'], filtered_df['Score 15aa'], label='15aa Score', marker='o')

    # Add titles and labels
    plt.title(f"Scores for {idr_name}")
    plt.xlabel('Residue Number')
    plt.ylabel('Score')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.show()

# Example usage:
# Plot for the IDR "A0A024RBG1_145_181" (specified range)
plot_idr_scores('A0A024RBG1_145_181')

# Plot for all residues starting with IDR "A0A024RBG1"
plot_idr_scores('A0A024RBG1')

# Plot for the specific residue range: "150-160"
plot_idr_scores('A0A024RBG1_150_160')

#%% baseline histogram of all residues
# get the max score for each residue
combined_per_res_scores['max_score'] = combined_per_res_scores[['Score 6aa', 'Score 10aa', 'Score 15aa']].max(axis=1)
# get the mean score for each residue
combined_per_res_scores['mean_score'] = combined_per_res_scores[['Score 6aa', 'Score 10aa', 'Score 15aa']].mean(axis=1)

# Plot a histogram of the maximum scores
plt.figure(figsize=(10, 6))
bins = np.linspace(0, 1, 101)
plt.hist(combined_per_res_scores['max_score'], bins=bins, color='skyblue', edgecolor='black')
plt.title('max res scores for all human IDRs',fontsize=14)
plt.xlabel('amyloidogenicity score',fontsize=14)
plt.ylabel('count',fontsize=14)
plt.xticks(fontsize=14) ; plt.yticks(fontsize=14)
# save png
plt.savefig('max_res_scores_histogram.png')

# plot a histogram of the mean scores
plt.figure(figsize=(10, 6))
plt.hist(combined_per_res_scores['mean_score'], bins=bins, color='skyblue', edgecolor='black')
plt.title('mean res scores for all human IDRs',fontsize=14)
plt.xlabel('amyloidogenicity score',fontsize=14)
plt.ylabel('count',fontsize=14)
plt.xticks(fontsize=14) ; plt.yticks(fontsize=14)
# save png
plt.savefig('mean_res_scores_histogram.png')
