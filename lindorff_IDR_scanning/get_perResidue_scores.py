#%%
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

def compute_per_residue_scores(scores_csv,scores_name='Score'):
    """
    Computes per-residue scores from fragment scores and saves the result as a CSV.
    
    Args:
    - scores_csv (str): The input CSV file containing fragment scores.
    """
    # Load the CSV file with fragment scores
    scores_df = pd.read_csv(scores_csv)

    # Dictionary to store per-residue scores
    residue_scores = defaultdict(dict)

    # Iterate through each row in the CSV file
    for index, row in tqdm(scores_df.iterrows(), total=len(scores_df)):
        fragment = row['Fragment']  # Example: "10-15|A0A075B6T7_1_32|A0A075B6T7"
        score = row['Score']  # The score corresponding to the fragment
        
        # Split the fragment identifier to get the start and end residues, and the IDR name
        fragment_info = fragment.split('|')
        residue_range = fragment_info[0]  # Example: "10-15"
        idr_name = fragment_info[1]  # Example: "A0A075B6T7_1_32"
        
        # Get the start and end positions of the fragment
        start_res, end_res = map(int, residue_range.split('-'))
        
        # Update per-residue scores, keeping the highest score for each residue
        for residue in range(start_res, end_res + 1):
            # Check if this residue already has a score and if the current score is higher
            if residue in residue_scores[idr_name]:
                residue_scores[idr_name][residue] = max(residue_scores[idr_name][residue], score)
            else:
                residue_scores[idr_name][residue] = score

    # Convert the residue scores into a DataFrame for easier export and analysis
    final_scores = []
    for idr_name, residues in residue_scores.items():
        for residue, score in residues.items():
            final_scores.append({'IDR': idr_name, 'Residue': residue, scores_name: score})

    # Convert to a DataFrame and save to CSV
    final_scores_df = pd.DataFrame(final_scores)
    return final_scores_df

# Apply the function to each of your scores CSVs
per_res_scores_6aa  = compute_per_residue_scores('scores_6aa.csv',scores_name='Score 6aa')
per_res_scores_10aa = compute_per_residue_scores('scores_10aa.csv', scores_name='Score 10aa')
per_res_scores_15aa = compute_per_residue_scores('scores_15aa.csv', scores_name='Score 15aa')

#%%
# Merge the DataFrames on 'IDR' and 'Residue' columns
combined_per_res_scores = pd.merge(per_res_scores_6aa, per_res_scores_10aa, on=['IDR', 'Residue'], how='outer')
combined_per_res_scores = pd.merge(combined_per_res_scores, per_res_scores_15aa, on=['IDR', 'Residue'], how='outer')
# save to pickled file
combined_per_res_scores.to_pickle('combined_per_res_scores.pkl')