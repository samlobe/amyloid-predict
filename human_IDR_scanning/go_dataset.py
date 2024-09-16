#%%
import pandas as pd
import pickle

df_GO_molfunc = pd.read_pickle('df_GO_molfunc.pkl')
df_GO_cellcom = pd.read_pickle('df_GO_cellcom.pkl')

# parse the fasta file
fasta = 'IDRs.fasta'

IDRs = []
with open(fasta) as file:
    for line in file:
        if line.startswith('>'):
            header = line.strip()
            IDRs.append(header[1:])

IDR_names = [IDR.split('|')[0] for IDR in IDRs]

# Initialize a list to store the GO terms for each protein
GO_terms = []

for IDR in IDR_names:
    # Initialize dictionary to hold cellcom and molfunc GO terms for this protein
    IDR_info = {'IDR': IDR, 'GO_molfunc': None, 'GO_cellcom': None}
    
    # Check if there is a GO term in the molfunc dataframe
    if IDR in df_GO_molfunc.index:
        IDR_info['GO_molfunc'] = df_GO_molfunc.loc[IDR, 'GO_terms']
    
    # Check if there is a GO term in the cellcom dataframe
    if IDR in df_GO_cellcom.index:
        IDR_info['GO_cellcom'] = df_GO_cellcom.loc[IDR, 'GO_terms']
    
    # Add the protein's info to the list
    GO_terms.append(IDR_info)

# Convert the list of dictionaries to a new DataFrame
df_GO_combined = pd.DataFrame(GO_terms)

# set IDR to be the index
df_GO_combined.set_index('IDR', inplace=True)

# Save the combined GO terms DataFrame to a pickle file
df_GO_combined.to_pickle('df_GO_combined.pkl')

#%% get unique terms
# For GO_molfunc
molfunc_terms = df_GO_combined['GO_molfunc'].dropna().explode().unique().tolist()
# For GO_cellcom
cellcom_terms = df_GO_combined['GO_cellcom'].dropna().explode().unique().tolist()
# alphabetically sort the terms
molfunc_terms.sort(); cellcom_terms.sort()

#%%
def find_idrs_by_molfunc(df, molfunc_term):
    # Find IDRs that have the molfunc_term in their GO_molfunc list
    with_molfunc = df[df['GO_molfunc'].apply(lambda x: molfunc_term in x if isinstance(x, list) else False)].index.tolist()
    
    # Find IDRs that do not have the molfunc_term in their GO_molfunc list
    without_molfunc = df[df['GO_molfunc'].apply(lambda x: molfunc_term not in x if isinstance(x, list) else True)].index.tolist()

    return with_molfunc, without_molfunc

def find_idrs_by_cellcom(df, cellcom_term):
    # Find IDRs that have the cellcom_term in their GO_cellcom list
    with_cellcom = df[df['GO_cellcom'].apply(lambda x: cellcom_term in x if isinstance(x, list) else False)].index.tolist()
    
    # Find IDRs that do not have the cellcom_term in their GO_cellcom list
    without_cellcom = df[df['GO_cellcom'].apply(lambda x: cellcom_term not in x if isinstance(x, list) else True)].index.tolist()

    return with_cellcom, without_cellcom

#%%
# Example usage
with_molfunc, without_molfunc = find_idrs_by_molfunc(df_GO_combined, 'actin binding')
with_cellcom, without_cellcom = find_idrs_by_cellcom(df_GO_combined, 'axon')