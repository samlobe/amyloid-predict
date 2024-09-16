#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined per-residue scores
combined_per_res_scores = pd.read_pickle('combined_per_res_scores.pkl')

#%%
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
    plt.scatter(filtered_df['Residue'], filtered_df['Score 6aa'], label='6aa Score', marker='o')
    plt.scatter(filtered_df['Residue'], filtered_df['Score 10aa'], label='10aa Score', marker='o')
    plt.scatter(filtered_df['Residue'], filtered_df['Score 15aa'], label='15aa Score', marker='o')

    # Add titles and labels
    plt.title(f"{idr_name}", fontsize=16)
    plt.xlabel('Residue Number', fontsize=14)
    plt.ylabel('Amyloidogenicity Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.show()

#%%Example usage:
# Plot for the IDR "A0A024RBG1_145_181" (specified range)
# plot_idr_scores('A0A024RBG1_145_181')

# Plot for all residues starting with IDR "A0A024RBG1"
plot_idr_scores('P10636_590_700') # tau
plot_idr_scores('Q9UBS5_1_80') # GABA B subunit 1
plot_idr_scores('Q9UBS5_800_1000') # GABA B subunit 1

# plot_idr_scores('O75899_1_50') # GABA B subunit 2
# plot_idr_scores('P47870_1_50') # GABA A
# plot_idr_scores('P47870') # GABA A

plot_idr_scores('P41594') # mGluR5

#%% Opsins
plot_idr_scores('P04000_1_50') # OPN1LW N-terminal
plot_idr_scores('P04000_325_375') # C-terminal
plot_idr_scores('P04001_1_50') # OPN1MW N-terminal
plot_idr_scores('P04001_325_375') # C-terminal
plot_idr_scores('P03999') # OPN1SW C-terminal
#%%
plot_idr_scores('P08100') # rhodopsin

#%% melanopsin
plot_idr_scores('Q9UHM6_1_75') # melanopsin N-terminal
plot_idr_scores('Q9UHM6_375_500') # melanopsin C-terminal

#%% encephalopsin
plot_idr_scores('Q9H1Y3_350_390') # encephalopsin C-terminal

#%% neuropsin
plot_idr_scores('Q6U736') # neuropsin C-terminal

#%% rgr opsin
# plot_idr_scores('P47804') 

#%% melanocyte stimulating hormone receptor
plot_idr_scores('Q01726')

#%% TSH receptor
plot_idr_scores('P16473_1_50')
plot_idr_scores('P16473_275_425')
plot_idr_scores('P16473_600_800')

#%% Calcium sensing receptor
plot_idr_scores('P41180_850_1100') # C-terminal

#%% CB1 receptor
plot_idr_scores('P21554_1_100') # N-terminal
plot_idr_scores('P21554_400_500') # C-terminal

#%% Melanocortin receptor
plot_idr_scores('P32245')

#%% Follicle stimulating hormone receptor # related to PCOS
plot_idr_scores('P23945_600_700') 
plot_idr_scores('P23945') 

#%% LH receptor
plot_idr_scores('P22888_1_50') # N-terminal
plot_idr_scores('P22888_640_700') # C-terminal

#%% vasopressin receptor
plot_idr_scores('P37288_250_450') # N-terminal
plot_idr_scores('P30518')

#%% 
# dopamine receptor 3
plot_idr_scores('P35462')
# dopamine receptor 2
plot_idr_scores('P14416_225_360')
# dopamine receptor 1
plot_idr_scores('P21728_340_450')

#%% adenosine A2A receptor
plot_idr_scores('P29274') #
#%% serotonin receptor
plot_idr_scores('P28223') #
plot_idr_scores('P08908')

#%% muscarinic receptor
plot_idr_scores('P11229') # M1
plot_idr_scores('P08172') # M2

#%% beta adrenergic receptor
# plot_idr_scores('P07550') # beta 1 

#%% a2a adenosine receptor
plot_idr_scores('P29274')

#%% Formyl Peptide Receptors
plot_idr_scores('P25090') 


#%% growth factors
# nerve growth factor
plot_idr_scores('P01138') # NGF
plot_idr_scores('P23560') # BDNF
plot_idr_scores('P39905') # GDNF
plot_idr_scores('P05019') # IGF1
plot_idr_scores('P15692') # VEGF
plot_idr_scores('P01137_1_50') # tgfb1 n-terminus
#%%

plot_idr_scores('P01133_1030_1160') # 
#%%
plot_idr_scores('P01133_1_50') # EGF
plot_idr_scores('P01133_370_400') # EGF
plot_idr_scores('P01133_790_840') # EGF
plot_idr_scores('P01133_1025_1060') # EGF
#%% known functional amyloids
plot_idr_scores('P40967_1_50') # PMEL
plot_idr_scores('P40967_320_420') # PMEL
plot_idr_scores('P40967_600_690') # PMEL

#%% cpeb3
plot_idr_scores('Q8NE35') # CPEB3

#%% fxr1
plot_idr_scores('P51114') # FXR1

#%% ZP proteins
plot_idr_scores('P60852') # ZP1
plot_idr_scores('P60852_1_50') # 7-21, 10-24
plot_idr_scores('P60852_120_220') # 146-160, 149-153, 152-156, 155-169
# 191-205, 194-208, 197-211
plot_idr_scores('P60852_590_650') # 597-611, 600-614, 603-617, 606-620, 609-623, 612-626, 615-629, 618-632

#%%
plot_idr_scores('Q05996') # ZP2
plot_idr_scores('Q05996_1_70') # 19-33, 22-, 25-, 28-, 31-, 34-, 37-, 40-54
plot_idr_scores('Q05996_110_190') # 129-143, 132-, 135-, 138-, 141-, 144-, 147-, 150-, 153-, 156-, 159-173, 175-188
plot_idr_scores('Q05996_450_500') # 461-475, 464-, 467-, 470-, 473-, 476-490
plot_idr_scores('Q05996_640_680') # 648-662, 651-665, 654-668
plot_idr_scores('Q05996_710_750') # 711-725, 714-728, 717-731, 720-734, 723-737, 726-740, 729-743

#%%
plot_idr_scores('P21754') # ZP3
plot_idr_scores('P21754_1_50') # 3-17
plot_idr_scores('P21754_350_420') # 355-369
# 382-396, 385-399, 388-, 391-, 394-, 397-, 400-, 403-, 406-420

#%%
plot_idr_scores('Q12836') # ZP4
plot_idr_scores('Q12836_1_60') # 1-15, 4-18, 7-21, 10-24, 29-43, 32-46
plot_idr_scores('Q12836_420_500') # 459-463, 470-484, 473-487, 476-490, 479-493, 482-496, 485-299
plot_idr_scores('Q12836_500_540') # 500-514, 503-, 506-, 509-, 512-, 515-, 518-, 521-535

#%%
plot_idr_scores('Q02297') # ctnf

#%% chemokine receptor
plot_idr_scores('P41597') # CCR2
plot_idr_scores('P49238') # CX3CR1

#%% angiotensin receptor
plot_idr_scores('P50052_1_50') # N-terminal
plot_idr_scores('P50052_310_370') # C-terminal

# %%

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
