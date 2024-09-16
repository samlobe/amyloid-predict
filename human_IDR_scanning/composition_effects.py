#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

cellcom_compositions = pd.read_csv('cellcom_compositions.csv',index_col=0)
molfunc_compositions = pd.read_csv('molfunc_compositions.csv',index_col=0)

compositions = pd.read_csv('compositions.csv',index_col=0)
IDR_compositions = compositions['all_IDRs']

cellcom_hists = pd.read_csv('cellcom_histograms.csv',index_col=0)
molfunc_hists = pd.read_csv('molfunc_histograms.csv',index_col=0)

#%%
# extract the histogram for all IDRs
all_IDR_hist = cellcom_hists[cellcom_hists.index=='ALL_IDRs'].with_term_hist.values

cellcoms = cellcom_compositions.index
molfuncs = molfunc_compositions.index
bins = cellcom_hists[cellcom_hists.index=='ALL_IDRs'].bin_edge.values
# extract just the histogram for each category cellcom_hists and create a new dataframe
cellcom_hist_clean = {}
for cellcom in cellcoms:
    cellcom_hist_clean[cellcom] = cellcom_hists[cellcom_hists.index==cellcom].with_term_hist.values
cellcom_hist_clean = pd.DataFrame(cellcom_hist_clean).T
# add the bin edges to the dataframe columns
cellcom_hist_clean.columns = bins

molfunc_hist_clean = {}
for molfunc in molfuncs:
    molfunc_hist_clean[molfunc] = molfunc_hists[molfunc_hists.index==molfunc].with_term_hist.values
molfunc_hist_clean = pd.DataFrame(molfunc_hist_clean).T
molfunc_hist_clean.columns = bins

#%%
# sum bins to track categories of low, medium, and high scores
molfunc_low_med_high = pd.DataFrame()
molfunc_low_med_high['<0.15'] = molfunc_hist_clean.iloc[:, :15].sum(axis=1)
molfunc_low_med_high['>0.80'] = molfunc_hist_clean.iloc[:, -20:].sum(axis=1)
molfunc_low_med_high['>0.20 & <0.40'] = molfunc_hist_clean.iloc[:, 20:40].sum(axis=1)

cellcom_low_med_high = pd.DataFrame()
cellcom_low_med_high['<0.15'] = cellcom_hist_clean.iloc[:, :15].sum(axis=1)
cellcom_low_med_high['>0.80'] = cellcom_hist_clean.iloc[:, -20:].sum(axis=1)
cellcom_low_med_high['>0.20 & <0.40'] = cellcom_hist_clean.iloc[:, 20:40].sum(axis=1)

#%%
# make bar plots, sorting by the low molfunc
df = molfunc_low_med_high.sort_values('<0.15')
fig, ax = plt.subplots(figsize=(15, 5))
df['<0.15'].plot(kind='bar', stacked=True, ax=ax)
plt.ylabel('% with score < 0.15', fontsize=14)
plt.yticks(fontsize=14)

df_ = molfunc_low_med_high.sort_values('>0.80')
fig, ax = plt.subplots(figsize=(15, 5))
df_['>0.80'].plot(kind='bar', stacked=True, ax=ax)
plt.ylabel('% with score > 0.80', fontsize=14)
plt.yticks(fontsize=14)

df__ = molfunc_low_med_high.sort_values('>0.20 & <0.40')
fig, ax = plt.subplots(figsize=(15, 5))
df__['>0.20 & <0.40'].plot(kind='bar', stacked=True, ax=ax)
plt.ylabel('% with score > 0.20 & < 0.40', fontsize=14)
plt.yticks(fontsize=14)

#%% do the same for cellcom
df = cellcom_low_med_high.sort_values('<0.15')
fig, ax = plt.subplots(figsize=(15, 5))
df['<0.15'].plot(kind='bar', stacked=True, ax=ax)
plt.ylabel('% with score < 0.15', fontsize=14)
plt.yticks(fontsize=14)

df_ = cellcom_low_med_high.sort_values('>0.80')
fig, ax = plt.subplots(figsize=(15, 5))
df_['>0.80'].plot(kind='bar', stacked=True, ax=ax)
plt.ylabel('% with score > 0.80', fontsize=14)
plt.yticks(fontsize=14)

df__ = cellcom_low_med_high.sort_values('>0.20 & <0.40')
fig, ax = plt.subplots(figsize=(15, 5))
df__['>0.20 & <0.40'].plot(kind='bar', stacked=True, ax=ax)
plt.ylabel('% with score > 0.20 & < 0.40', fontsize=14)
plt.yticks(fontsize=14)

#%%
def calculate_kl_divergence(aa_probs, full_idr_probs):
    """
    Calculates the Kullback-Leibler (KL) divergence between amino acid compositions of a GO category
    and the full IDR amino acid composition.
    
    Args:
    - aa_probs (dict): Amino acid probabilities for the GO term.
    - full_idr_probs (dict): Full IDR amino acid probabilities.
    
    Returns:
    - kl_divergence (float): KL divergence between the two compositions.
    """
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_probs_list = [aa_probs[aa] for aa in aa_list]
    full_idr_probs_list = [full_idr_probs[aa] for aa in aa_list]
    
    # Ensure there are no zero values to avoid log(0) issues
    aa_probs_list = [max(p, 1e-10) for p in aa_probs_list]
    full_idr_probs_list = [max(p, 1e-10) for p in full_idr_probs_list]
    
    return entropy(aa_probs_list, full_idr_probs_list)

# Calculate KL divergence for molfunc_compositions
kl_divergences = []
for index, row in molfunc_compositions.iterrows():
    aa_probs = row.to_dict()  # Get amino acid probabilities for this GO term
    kl_div = calculate_kl_divergence(aa_probs, IDR_compositions)
    kl_divergences.append(kl_div)

# Add the KL divergences to molfunc_compositions
molfunc_compositions['kl_divergence'] = kl_divergences

# Merge the KL divergence and the amyloidogenicity percentages into a single DataFrame
molfunc_scatter_data = pd.DataFrame({
    'kl_divergence': molfunc_compositions['kl_divergence'],
    '>0.80': molfunc_low_med_high['>0.80']
})

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(molfunc_scatter_data['kl_divergence'], molfunc_scatter_data['>0.80'], color='blue', alpha=0.7)
plt.xlabel('KL Divergence (Amino Acid Composition Deviation)', fontsize=14)
plt.ylabel('High Amyloidogenicity Percentage (>0.80)', fontsize=14)
plt.title('AA Composition Divergence vs High Amyloidogenicity Percentage', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)

# annotate the text above the points with y-axis>3 or x-axis>0.02
for i, txt in enumerate(molfunc_scatter_data.index):
    if molfunc_scatter_data['>0.80'][i] > 3 or molfunc_scatter_data['kl_divergence'][i] > 0.02:
        print(txt)
        if txt == "integrin binding":
            plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i], molfunc_scatter_data['>0.80'][i]-0.2), fontsize=12)
        elif txt == "extracellular matrix structural constituent":
            plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i]-0.05, molfunc_scatter_data['>0.80'][i]+0.2), fontsize=12)
        elif txt == "heparin binding":
            plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i]+0.002, molfunc_scatter_data['>0.80'][i]-0.1), fontsize=12)
        elif txt == "calcium ion binding":
            plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i], molfunc_scatter_data['>0.80'][i]-0.2), fontsize=12)
        else:
            plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i], molfunc_scatter_data['>0.80'][i]), fontsize=12)

plt.show()

#%% do the same for low amyloidogenicity
molfunc_scatter_data = pd.DataFrame({
    'kl_divergence': molfunc_compositions['kl_divergence'],
    '<0.15': molfunc_low_med_high['<0.15']
})

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(molfunc_scatter_data['kl_divergence'], molfunc_scatter_data['<0.15'], color='blue', alpha=0.7)
plt.xlabel('KL Divergence (Amino Acid Composition Deviation)', fontsize=14)
plt.ylabel('Low Amyloidogenicity Percentage (<0.15)', fontsize=14)
plt.title('AA Composition Divergence vs Low Amyloidogenicity Percentage', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)

# annotate the text above the points with y-axis>3 or x-axis>0.02
for i, txt in enumerate(molfunc_scatter_data.index):
    if molfunc_scatter_data['<0.15'][i] > 16 or molfunc_scatter_data['kl_divergence'][i] > 0.02:
        print(txt)
        if txt == "extracellular matrix structural constituent":
            plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i]-0.05, molfunc_scatter_data['<0.15'][i]+0.2), fontsize=12)
        # elif txt == "calcium ion binding":
        #     plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i], molfunc_scatter_data['<0.15'][i]-0.2), fontsize=12)
        # else:
        else: plt.annotate(txt, (molfunc_scatter_data['kl_divergence'][i], molfunc_scatter_data['<0.15'][i]), fontsize=12)

#%%
# do the same for cellcom
kl_divergences = []
for index, row in cellcom_compositions.iterrows():
    aa_probs = row.to_dict()  # Get amino acid probabilities for this GO term
    kl_div = calculate_kl_divergence(aa_probs, IDR_compositions)
    kl_divergences.append(kl_div)

# Add the KL divergences to molfunc_compositions
cellcom_compositions['kl_divergence'] = kl_divergences

# Merge the KL divergence and the amyloidogenicity percentages into a single DataFrame
cellcom_scatter_data = pd.DataFrame({
    'kl_divergence': cellcom_compositions['kl_divergence'],
    '>0.80': cellcom_low_med_high['>0.80']
})

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(cellcom_scatter_data['kl_divergence'], cellcom_scatter_data['>0.80'], color='blue', alpha=0.7)
plt.xlabel('KL Divergence (Amino Acid Composition Deviation)', fontsize=14)
plt.ylabel('High Amyloidogenicity Percentage (>0.80)', fontsize=14)
plt.title('AA Composition Divergence vs High Amyloidogenicity Percentage', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)

# annotate the text above the points with y-axis>3 or x-axis>0.02
for i, txt in enumerate(cellcom_scatter_data.index):
    if cellcom_scatter_data['>0.80'][i] > 3 or cellcom_scatter_data['kl_divergence'][i] > 0.02:
        print(txt)
        if txt == "collagen-containing extracellular matrix":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.005, cellcom_scatter_data['>0.80'][i]-0.25), fontsize=12)
        elif txt == "endoplasmic reticulum lumen":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.02, cellcom_scatter_data['>0.80'][i]+0.2), fontsize=12)
        elif txt == "Golgi lumen":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.01, cellcom_scatter_data['>0.80'][i]+0.2), fontsize=12)
        else:
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i], cellcom_scatter_data['>0.80'][i]), fontsize=12)

plt.show()
#%% finally for low amyloidogenicity cellcom
cellcom_scatter_data = pd.DataFrame({
    'kl_divergence': cellcom_compositions['kl_divergence'],
    '<0.15': cellcom_low_med_high['<0.15']
})

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(cellcom_scatter_data['kl_divergence'], cellcom_scatter_data['<0.15'], color='blue', alpha=0.7)
plt.xlabel('KL Divergence (Amino Acid Composition Deviation)', fontsize=14)
plt.ylabel('Low Amyloidogenicity Percentage (<0.15)', fontsize=14)
plt.title('AA Composition Divergence vs Low Amyloidogenicity Percentage', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)

# annotate the text above the points with y-axis>3 or x-axis>0.02
for i, txt in enumerate(cellcom_scatter_data.index):
    if cellcom_scatter_data['<0.15'][i] > 16 or cellcom_scatter_data['kl_divergence'][i] > 0.02:
        print(txt)
        if txt == "collagen-containing extracellular matrix":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.005, cellcom_scatter_data['<0.15'][i]+0.3), fontsize=12)
        elif txt == "extracellular matrix":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.012, cellcom_scatter_data['<0.15'][i]-0.6), fontsize=12)
        elif txt == "Golgi lumen":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.012, cellcom_scatter_data['<0.15'][i]-0.6), fontsize=12)
        elif txt == "endoplasmic reticulum lumen":
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i]-0.019, cellcom_scatter_data['<0.15'][i]+0.2), fontsize=12)
        else:
            plt.annotate(txt, (cellcom_scatter_data['kl_divergence'][i], cellcom_scatter_data['<0.15'][i]), fontsize=12)

#%% plot the differences in amino acid composition for a few selected GO terms
molfunc_terms_to_plot = ['RNA binding','mRNA binding']
# molfunc_terms_to_plot = ['carbohydrate binding','growth factor activity','transmembrane signaling receptor activity',
#                          'signaling receptor activity', 'metalloendopeptidase activity','G protein-coupled receptor activity',
#                          'heparin binding','protein tyrosine kinase activity','calcium ion binding',
#                          'integrin binding']
# molfunc_terms_to_plot = ['actin filament binding','microtubule binding','extracellular matrix structural constituent']
for term in molfunc_terms_to_plot:
    aa_probs = molfunc_compositions.loc[term].loc['A':'Y']
    full_idr_probs = IDR_compositions.loc['A':'Y']
    diff = aa_probs - full_idr_probs
    plt.figure(figsize=(8, 12))
    colors = ['tab:orange' if x < 0 else 'gray' for x in diff]
    plt.bar(diff.index, diff, color=colors)
    plt.title(f'{term} - All IDRs', fontsize=16)
    plt.ylabel("Frequncy Difference", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-0.04,0.04)

#%% do the same for cellcom
cellcom_terms_to_plot = ['external side of plasma membrane','cell surface','receptor complex',
                         'Golgi membrane','extracellular region','extracellular space']
cellcom_terms_to_plot = ['nuclear speck','ribonucleoprotein complex']
cellcom_terms_to_plot = ['Golgi lumen','basement membrane','endoplasmic reticulum lumen',
                         'collagen-containing extracellular matrix','extracellular matrix',
                         'mitochondrial matrix','mitochondrial inner membrane']
for term in cellcom_terms_to_plot:
    aa_probs = cellcom_compositions.loc[term].loc['A':'Y']
    full_idr_probs = IDR_compositions.loc['A':'Y']
    diff = aa_probs - full_idr_probs
    plt.figure(figsize=(8, 6))
    colors = ['tab:orange' if x < 0 else 'gray' for x in diff]
    plt.bar(diff.index, diff, color=colors)
    plt.title(f'{term} - All IDRs', fontsize=16)
    plt.ylabel("Frequncy Difference", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-0.04,0.04)