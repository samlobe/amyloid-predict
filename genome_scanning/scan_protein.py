#%%
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_models(model_dir):
    """
    Load the classification models and selected features for the ESM2-3B embeddings.
    
    Args:
    - model_dir (str): The directory containing the model files.
    
    Returns:
    - model_6aa (joblib): The model trained on 6aa peptides from WALTZdb.
    - model_10aa (joblib): The model trained on 10aa fragments of PrP, lysozyme, and β-microglobulin from Fernandez-Escamilla, et. al. 2004 (TANGO paper).
    - model_15aa (joblib): The model trained on 15aa tau fragments from Louros et. al. 2024 (PAM4 paper).
    - ensemble_model (joblib): The ensemble model that uses logits from the other 3 models for the final prediction.
    - selected_features (array): The selected features used for training the models.
    """
    model_6aa = joblib.load(f'{model_dir}/WALTZtht_top_model.joblib')
    model_10aa = joblib.load(f'{model_dir}/TANGO_Table2_top_model.joblib')
    model_15aa = joblib.load(f'{model_dir}/tau_top_model.joblib')
    ensemble_model = joblib.load(f'{model_dir}/ensemble_model.joblib')
    selected_features = np.loadtxt(f'{model_dir}/selected_features_3B.csv', dtype=str)
    
    return model_6aa, model_10aa, model_15aa, ensemble_model, selected_features

def load_embeddings(protein_name, embeddings_dir='human_genome_embeddings', selected_features=None):
    """
    Load the ESM2-3B embeddings for a given protein and return a DataFrame of selected features.
    
    Args:
    - protein_name (str): The full name of the protein in the format 'sp|ID|PROTEIN_NAME OS=...'
    - embeddings_dir (str): The directory containing the embeddings files.
    - selected_features (list or array): A list of selected feature names to filter the embeddings.
    
    Returns:
    - selected_embeddings_df (pd.DataFrame): DataFrame containing selected embeddings for the protein.
    """
    embeddings_file = f'{embeddings_dir}/{protein_name}.pt'
    
    # Read the embeddings
    embeddings = torch.load(embeddings_file)['representations'][36].numpy()
    
    # Turn embeddings into a DataFrame
    columns = [f'embedding_{i}' for i in range(2560)]
    index = np.arange(1, len(embeddings) + 1)  # residue id
    embeddings_df = pd.DataFrame(embeddings, columns=columns, index=index)
    
    # Select specific features if provided
    if selected_features is not None:
        selected_embeddings_df = embeddings_df[selected_features]
    else:
        selected_embeddings_df = embeddings_df
    
    return selected_embeddings_df

def generate_fragments_and_avg_embeddings(embeddings_df, fragment_length):
    avg_frag_embeddings = []
    residue_ranges = []
    
    # Loop through each fragment of the specified length
    for i in range(len(embeddings_df) - fragment_length + 1):
        frag_embeddings = embeddings_df.iloc[i:i+fragment_length]
        
        # Calculate the average embedding for the fragment
        avg_embedding = frag_embeddings.mean(axis=0)
        avg_frag_embeddings.append(avg_embedding)
        
        # Create a residue range for labeling (e.g., "1-6", "2-7")
        residue_range = f"{i+1}-{i+fragment_length}"
        residue_ranges.append(residue_range)
    
    # Convert the list of average embeddings into a DataFrame
    avg_frag_embeddings_df = pd.DataFrame(avg_frag_embeddings)
    
    # Add the residue ranges as the index
    avg_frag_embeddings_df['residue_range'] = residue_ranges
    avg_frag_embeddings_df.set_index('residue_range', inplace=True)
    
    return avg_frag_embeddings_df

# Function to predict amyloidogenicity scores
def predict_scores(model, avg_frag_embeddings_df):
    scores = model.predict_proba(avg_frag_embeddings_df)[:, 1]  # probability of being amyloidogenic
    # set the index to the residue ranges
    scores = pd.Series(scores, index=avg_frag_embeddings_df.index)
    return scores

def compute_per_residue_scores(scores):
    # get the fragment length from the last index
    fragment_length = int(scores.index[-1].split('-')[1]) - int(scores.index[-1].split('-')[0]) + 1
    # get the number of residues from the last index
    num_residues = int(scores.index[-1].split('-')[1])
    
    # Initialize an array to store the maximum score for each residue
    per_residue_scores = np.zeros(num_residues)
    
    # Loop through each fragment
    for i, fragment_range in enumerate(scores.index):
        score = scores.iloc[i]
        
        # Extract the start and end residue indices from the fragment range
        start_residue = int(fragment_range.split('-')[0])  # Starting residue
        end_residue = int(fragment_range.split('-')[1])    # Ending residue
        
        # Update the score for each residue in the fragment range
        for res in range(start_residue, end_residue + 1):
            per_residue_scores[res - 1] = max(per_residue_scores[res - 1], score)  # Update with the max score
    
    return pd.Series(per_residue_scores, index=np.arange(1, num_residues + 1))

def plot_scores(scores, protein_name):
    plt.figure(figsize=(10, 6))  # Set a suitable figure size
    plt.bar(scores.index, scores.values, color='b')
    plt.ylabel('Amyloidogenicity')
    plt.xlabel('Residue ID')
    plt.xticks(rotation=90)  # Rotate x-ticks for better visibility
    plt.title(f'{protein_name}')
    plt.tight_layout()
    plt.show()

def scan_protein(protein_name, model_6aa, model_10aa, model_15aa, selected_features, embeddings_dir='human_genome_embeddings'):
    # Load the embeddings
    selected_embeddings_df = load_embeddings(protein_name, embeddings_dir=embeddings_dir, selected_features=selected_features)
    
    # Get the number of residues in the protein
    num_residues = len(selected_embeddings_df)
    
    # Initialize empty scores DataFrame
    scores_df = pd.DataFrame(index=np.arange(1, num_residues + 1))
    
    # Generate 6aa fragments, average their embeddings, score the fragments, and get per-residue scores
    if num_residues >= 6:
        avg_6aa_fragments_df = generate_fragments_and_avg_embeddings(selected_embeddings_df, fragment_length=6)
        frag_scores_6aa = predict_scores(model_6aa, avg_6aa_fragments_df)
        scores_6aa = compute_per_residue_scores(frag_scores_6aa)
        scores_df['6aa'] = scores_6aa
    else:
        print(f"Protein {protein_name} is too short for 6aa model")
        scores_df['6aa'] = np.nan  # Mark as NaN if too short
    
    # Generate 10aa fragments, average their embeddings, score the fragments, and get per-residue scores
    if num_residues >= 10:
        avg_10aa_fragments_df = generate_fragments_and_avg_embeddings(selected_embeddings_df, fragment_length=10)
        frag_scores_10aa = predict_scores(model_10aa, avg_10aa_fragments_df)
        scores_10aa = compute_per_residue_scores(frag_scores_10aa)
        scores_df['10aa'] = scores_10aa
    else:
        print(f"Protein {protein_name} is too short for 10aa model")
        scores_df['10aa'] = np.nan  # Mark as NaN if too short
    
    # Generate 15aa fragments, average their embeddings, score the fragments, and get per-residue scores
    if num_residues >= 15:
        avg_15aa_fragments_df = generate_fragments_and_avg_embeddings(selected_embeddings_df, fragment_length=15)
        frag_scores_15aa = predict_scores(model_15aa, avg_15aa_fragments_df)
        scores_15aa = compute_per_residue_scores(frag_scores_15aa)
        scores_df['15aa'] = scores_15aa
    else:
        print(f"Protein {protein_name} is too short for 15aa model")
        scores_df['15aa'] = np.nan  # Mark as NaN if too short

    return scores_df


#%%
if __name__ == '__main__':
    # Pick a protein
    protein_name = 'sp|P10636|TAU_HUMAN Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5'
    protein_name = 'sp|P10997|IAPP_HUMAN Islet amyloid polypeptide OS=Homo sapiens OX=9606 GN=IAPP PE=1 SV=1'
    protein_name = 'sp|P0DJI8|SAA1_HUMAN Serum amyloid A-1 protein OS=Homo sapiens OX=9606 GN=SAA1 PE=1 SV=2'
    protein_name = 'sp|P37840|SYUA_HUMAN Alpha-synuclein OS=Homo sapiens OX=9606 GN=SNCA PE=1 SV=1'
    # protein_name = 'sp|Q9Y6H3|ATP23_HUMAN Mitochondrial inner membrane protease ATP23 homolog OS=Homo sapiens OX=9606 GN=ATP23 PE=1 SV=3'
    # protein_name = 'tr|Q6JHZ5|Q6JHZ5_HUMAN NS5ATP13TP1 OS=Homo sapiens OX=9606 PE=2 SV=1'
    protein_name = 'sp|A0A0A0MT89|KJ01_HUMAN Immunoglobulin kappa joining 1 OS=Homo sapiens OX=9606 GN=IGKJ1 PE=4 SV=2'

    # load models
    model_6aa, model_10aa, model_15aa, ensemble_model, selected_features = load_models('../model_development/models_3B')
    # scan protein
    scores_df = scan_protein(protein_name, model_6aa, model_10aa, model_15aa, selected_features, embeddings_dir='human_genome_embeddings')
    scores_6aa = scores_df['6aa']; scores_10aa = scores_df['10aa']; scores_15aa = scores_df['15aa']

    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(scores_6aa.index, scores_6aa.values, color='b', label='6aa frag model')
    plt.plot(scores_10aa.index, scores_10aa.values, color='r', label='10aa frag model')
    plt.plot(scores_15aa.index, scores_15aa.values, color='g', label='15aa frag model')
    plt.ylabel('Amyloidogenicity', fontsize=14)
    plt.xlabel('Residue ID', fontsize=14)
    protein_short_name = protein_name.split('|')[2].split(' OS=')[0]
    plt.title(protein_short_name, fontsize=16)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=14)
# %%
