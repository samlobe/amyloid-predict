#%%
import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    
    return model_6aa, model_10aa, model_15aa, selected_features

def load_embeddings(IDR_frag_name, embeddings_dir, selected_features):
    """
    Load the ESM2-3B embeddings for a given protein and return a DataFrame of selected features.
    
    Args:
    - IDR_frag_name (str): The full name of the protein in the format 'starti-endi|name|gene'
    - embeddings_dir (str): The directory containing the embeddings files.
    - selected_features (list or array): A list of selected feature names to filter the embeddings.
    
    Returns:
    - selected_embeddings_df (pd.DataFrame): DataFrame containing selected embeddings for the protein.
    """
    embeddings_file = f'{embeddings_dir}/{IDR_frag_name}.pt'
    
    # Read the embeddings
    embeddings = torch.load(embeddings_file)['mean_representations'][36].numpy()
    
    # Turn embeddings into a DataFrame
    index = [f'embedding_{i}' for i in range(2560)]
    embeddings_series = pd.Series(embeddings, index=index)
    
    # Select specific features for the model
    selected_embeddings_series = embeddings_series[selected_features]
    
    return selected_embeddings_series

# Function to predict amyloidogenicity scores
def predict_score(model, embeddings_series):
    embeddings_df = embeddings_series.values.reshape(1, -1)
    score = model.predict_proba(embeddings_df)[:, 1]  # probability of being amyloidogenic
    # set the index to the residue ranges
    return float(score[0])

def parse_fasta(fasta_file):
    """
    Parse a FASTA file and return a list of headers and a list of sequences.
    
    Args:
    - fasta_file (str): The path to the FASTA file.
    
    Returns:
    - headers (list): A list of headers.
    """
    headers = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            if line[0] == '>':
                headers.append(line.strip()[1:])  # Remove the '>' from the header
    return headers

#%%

# load models
model_dir = '../model_development/models_3B'
model_6aa, model_10aa, model_15aa, selected_features = load_models(model_dir)

# parse fragment files
frags_6aa = parse_fasta("IDRs_6aa_frag_1sw.fasta")
frags_10aa = parse_fasta("IDRs_10aa_frag_2sw.fasta")
frags_15aa = parse_fasta("IDRs_15aa_frag_3sw.fasta")

#%%
scores_6aa = []
for frag in tqdm(frags_6aa):
    embeddings = load_embeddings(frag, 'IDRs_6aa_frag_1sw', selected_features)
    scores_6aa.append(predict_score(model_6aa, embeddings))
# %%
# Create a DataFrame to store fragment names and their scores
scores_df_6aa = pd.DataFrame({'Fragment': frags_6aa,'Score': scores_6aa})
scores_df_6aa.to_csv('scores_6aa.csv', index=False)
print("Scores saved to scores_6aa.csv")
# %%
scores_10aa = []
for frag in tqdm(frags_10aa):
    embeddings = load_embeddings(frag, 'IDRs_10aa_frag_2sw', selected_features)
    scores_10aa.append(predict_score(model_10aa, embeddings))
scores_df_10aa = pd.DataFrame({'Fragment': frags_10aa,'Score': scores_10aa})
scores_df_10aa.to_csv('scores_10aa.csv', index=False)
print("Scores saved to scores_10aa.csv")
#%%
scores_15aa = []
for frag in tqdm(frags_15aa):
    embeddings = load_embeddings(frag, 'IDRs_15aa_frag_3sw', selected_features)
    scores_15aa.append(predict_score(model_15aa, embeddings))
scores_df_15aa = pd.DataFrame({'Fragment': frags_15aa,'Score': scores_15aa})
scores_df_15aa.to_csv('scores_15aa.csv', index=False)
print("Scores saved to scores_15aa.csv")
# %%
