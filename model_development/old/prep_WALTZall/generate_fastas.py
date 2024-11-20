#%%
import pandas as pd

# read the ThT_Binding data from WALTZdb
# this is just the data that had yes or no for ThT binding
df = pd.read_csv('../WALTZall_labels.csv')

peptides = df['Sequence'].values # dang there's repeats, let me fix that

# get a set of the unique peptides
unique_peptides = set(peptides)

# make a new df with the unique peptides and their classification
# first make a dictionary with the unique peptides and their classification
d = {}
for peptide in unique_peptides:
    d[peptide] = df[df['Sequence'] == peptide]['Classification'].values[0]

# then make a new df
df = pd.DataFrame(d.items(), columns=['Sequence', 'Classification'])
peptides = df['Sequence'].values
#%%
with open('WALTZall.fasta', 'w') as f:
    for peptide in peptides:
        f.write('>' + peptide + '\n')
        f.write(peptide + '\n')
        # for i in range(n_peptides-1):
        #     f.write(peptide + 'UUUUUUUUUU')
        # f.write(peptide + '\n')