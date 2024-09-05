#%%
from scan_protein import scan_protein
from scan_protein import load_models
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# list all the proteins in the embeddings file
embeddings_dir = 'human_genome_embeddings'
embeddings_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.pt')]
protein_names = [f.split('.pt')[0] for f in embeddings_files]
# sort the proteins by name
protein_names.sort()
protein_names_short = [protein_name.split('|')[2].split(' OS=')[0] for protein_name in protein_names]

# load the models
model_dir = '../model_development/models_3B'
model_6aa, model_10aa, model_15aa, ensemble_model, selected_features = load_models(model_dir)

# initialize a list to score histograms
score_histograms = []
# the histogram will have 100 bins from 0 to 1
bins = np.linspace(0, 1, 101)

# scan each protein
output_dir = 'human_genome_res_max_scores'
os.makedirs(output_dir, exist_ok=True)
for protein in tqdm(protein_names):
    # print(f'Scanning {protein}')
    _,_,_,scores_df = scan_protein(protein, model_6aa, model_10aa, model_15aa, selected_features, embeddings_dir=embeddings_dir)
    # save a csv filr of the scores df
    scores_df.to_csv(f'{output_dir}/{protein}.csv')
    # get max score for each index
    max_scores = scores_df.max(axis=1)
    # get histogram bin heights of max scores
    hist, _ = np.histogram(max_scores, bins=bins)
    score_histograms.append(hist)

#%%
# turn the histograms into a DataFrame where each row is a protein and each column is a bin
score_histograms_df = pd.DataFrame(score_histograms, columns=bins[1:],index=protein_names)
# make sure each column only has two decimal places
score_histograms_df.columns = score_histograms_df.columns.round(2)
score_histograms_df.to_csv('human_genome_res_max_scores_histograms.csv')
