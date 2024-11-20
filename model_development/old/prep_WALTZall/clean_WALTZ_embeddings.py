#%%
import torch
import pandas as pd

esm15B_dir = 'esm2_15B_output'
esm3B_dir = 'esm2_3B_output'

df = pd.read_csv('waltzdb_export.csv')

sequences = df['Sequence'].values
# remove the non-unique sequences rows
df = df.drop_duplicates(subset='Sequence', keep='first')
# reindex
df = df.reset_index(drop=True)

# count how many of the "Th-T Binding" columns is NaN
print("How many don't have a Th-T Binding value?")
print(df['Th-T Binding'].isna().sum())

# set the Sequence column as the index
df = df.set_index('Sequence')

# get the indices of the NaN values
nan_indices = df[df['Th-T Binding'].isna()].index

# create a new df, pulling out the rows where index=nan_indices
df_nan = df.loc[nan_indices]

# pull out just the classification column
y = df_nan['Classification']
# turn the non-amyloid to 0 and amyloid to 1
# y = y.map({'non-amyloid':0, 'amyloid':1})

# save this as a csv: WALTZall_labels.csv
y.to_csv('WALTZall_labels.csv')

#%%

esm3B_embeddings = []
esm15B_embeddings = []

for sequence in tqdm(nan_indices):
    # load the embeddings
    esm3B_embedding = torch.load(f'{esm3B_dir}/{sequence}.pt')['mean_representations'][36].numpy()
    esm15B_embedding = torch.load(f'{esm15B_dir}/{sequence}.pt')['mean_representations'][48].numpy()
    esm3B_embeddings.append(esm3B_embedding)
    esm15B_embeddings.append(esm15B_embedding)

esm3B_embeddings = np.array(esm3B_embeddings)
esm15B_embeddings = np.array(esm15B_embeddings)

# turn into df
df3B = pd.DataFrame(esm3B_embeddings, index=nan_indices)
df15B = pd.DataFrame(esm15B_embeddings, index=nan_indices)

# assign df3B columns to be embedding_0 to embedding_2559
# assign df15B columns to be embedding_0 to embedding_5119
df3B.columns = [f'embedding_{i}' for i in range(2560)]
df15B.columns = [f'embedding_{i}' for i in range(5120)]

# save the embeddings
df3B.to_csv('WALTZall_3B_embeddings.csv')
df15B.to_csv('WALTZall_15B_embeddings.csv')