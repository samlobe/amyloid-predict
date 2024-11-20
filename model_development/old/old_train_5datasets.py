#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

#%% import features from esm_3B and labels
def load_data(dataset,esm_model,tau_amyloid_def='either',WALTZ_amyloid_def='Classification'):
    embeddings = pd.read_csv(f'features/{dataset}_{esm_model}_embeddings.csv',index_col=0)
    if dataset == 'tau':
        y = pd.read_csv('labels/tau_labels.csv',index_col=0)[tau_amyloid_def]
    elif dataset == 'WALTZall': # all the WALTZ data without Th-T Binding values
        y = pd.read_csv('labels/WALTZall_labels.csv',index_col=0)['Classification']
        y = y.map({'amyloid':True,'non-amyloid':False})
    elif dataset == 'WALTZtht': # only the WALTZ data with Th-T Binding values
        y = pd.read_csv('labels/WALTZtht_labels.csv',index_col=0)[WALTZ_amyloid_def]
        y = y.map({'amyloid':True,'non-amyloid':False})
    elif dataset == 'TANGO_Table1':
        y = pd.read_csv('labels/TANGO_Table1_labels.csv',index_col=0)['Experimental Aggregation Behavior']
        y = y.map({'+':True,'-':False})
    elif dataset == 'TANGO_Table2':
        y = pd.read_csv('labels/TANGO_Table2_labels.csv',index_col=0)['Experimental Aggregation Behavior']
        y = y.map({'+':True,'-':False})
    # rename y-column to 'amyloid'
    y = y.rename('amyloid')

    # merge embeddings and labels
    data = pd.concat([y,embeddings],axis=1)
    return data

def load_all_data(esm_model,tau_amyloid_def='either',WALTZ_amyloid_def='Classification'):
    tau_data = load_data('tau',esm_model,tau_amyloid_def,WALTZ_amyloid_def)
    WALTZall_data = load_data('WALTZall',esm_model,tau_amyloid_def,WALTZ_amyloid_def)
    WALTZtht_data = load_data('WALTZtht',esm_model,tau_amyloid_def,WALTZ_amyloid_def)
    TANGO_Table1_data = load_data('TANGO_Table1',esm_model)
    TANGO_Table2_data = load_data('TANGO_Table2',esm_model)
    # merge datasets while labeling the source
    tau_data['dataset'] = 'tau'
    WALTZall_data['dataset'] = 'WALTZall'
    WALTZtht_data['dataset'] = 'WALTZtht'
    TANGO_Table1_data['dataset'] = 'TANGO_Table1'
    TANGO_Table2_data['dataset'] = 'TANGO_Table2'
    data = pd.concat([tau_data,WALTZall_data,WALTZtht_data,TANGO_Table1_data,TANGO_Table2_data])
    # set index to 'dataset' and 'index'
    data = data.set_index(['dataset',data.index])
    return data

#%%
# load data for 3B model
esm_model = '3B' # 3B or 15B
tau_amyloid_def = 'either' # ThT fluorescence or pFTAA fluorescene or either (at least 50% more fluorescence than bulk)
WALTZ_amyloid_def = 'Classification' # "Classification" vs "Th-T Binding"

data = load_all_data(esm_model,tau_amyloid_def,WALTZ_amyloid_def)
# X = data.drop('amyloid',axis=1)
# X = X.droplevel(0) # removing the dataset index from X

# y = data['amyloid']
# y = y.droplevel(0) # remove the dataset index from y

# get a list of the datasets: tau, WALTZall, WALTZtht, TANGO_Table1, TANGO_Table2
datasets = data.index.get_level_values(0).unique()

#%% do logistic regression with l1 regularization on each dataset
# tune the regularization strength to maximize avg AUC across all datasets
# use balanced class weights
# save the features for the best model for each dataset

# Initialize parameters
C_values = np.logspace(-2, 2, 8)
best_features = []
avg_auc_lists = []  # Store avg AUCs for plotting
        
# Loop through each dataset
for dataset in datasets:
    print(f"Training on dataset: {dataset}")
    X_train = data.loc[dataset].drop('amyloid', axis=1)
    y_train = data.loc[dataset]['amyloid']

    best_avg_auc = -np.inf
    best_C = None
    best_features_for_dataset = None
    avg_aucs = []

    for C in tqdm(C_values):
        # Using LogisticRegression with L1 regularization
        logistic_reg = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=C, 
            max_iter=10000, 
            class_weight='balanced'
        )
        logistic_reg.fit(X_train, y_train)

        # Validate on other datasets
        aucs = []
        for vDataset in datasets:
            X_val = data.loc[vDataset].drop('amyloid', axis=1)
            y_val = data.loc[vDataset]['amyloid']
            y_pred = logistic_reg.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            aucs.append(auc)
        avg_auc = np.mean(aucs)
        avg_aucs.append(avg_auc)

        # Check if this model has the best average AUC
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_C = C
            best_features_for_dataset = X_train.columns[logistic_reg.coef_[0] != 0]

    # Store the best features for this dataset
    if best_features_for_dataset is not None:
        best_features.append(list(best_features_for_dataset))

    # Store the avg AUCs for plotting
    avg_auc_lists.append(avg_aucs)

#%%
# Plot C vs avg ROC-AUC for each dataset
plt.figure(figsize=(10, 6))
for i, dataset in enumerate(datasets):
    plt.plot(C_values, avg_auc_lists[i], label=f'Dataset: {dataset}')
plt.xscale('log')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Average ROC-AUC')
plt.title('C vs Average ROC-AUC for 5 Datasets')
plt.legend()
plt.show()

# Output the final set of selected features for each dataset
for i, dataset in enumerate(datasets):
    # print(f"Selected features for dataset {dataset}: {best_features[i]}")
    print(len(best_features[i]))

# print the set of features that are selected in at least one dataset
selected_features = set()
for features in best_features:
    selected_features.update(features)
selected_features = list(selected_features)
print(len(selected_features))

# turn selected features into a dictionary with the dataset as the key
dataset_features = {dataset: [] for dataset in datasets}
for i, dataset in enumerate(datasets):
    dataset_features[dataset] = best_features[i]


#%%
# Initialize parameters
C_values = np.logspace(-0.5, 4, 8)
l1_ratios = [0, 0.5]
models = {'LR': [], 'SVM': [], 'LDA': []}
roc_auc_results = pd.DataFrame(index=datasets, columns=datasets)  # DataFrame to store results
avg_auc_lists = {}

# Train models using selected features
for dataset_idx, dataset in enumerate(datasets):
    X_train = data.loc[dataset].drop('amyloid', axis=1)[dataset_features[dataset]]
    y_train = data.loc[dataset]['amyloid']

    best_model = None
    best_avg_auc = -np.inf
    best_params = None

    # Logistic Regression with Elastic Net
    for l1_ratio in l1_ratios:
        for C in tqdm(C_values):
            logistic_reg = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=l1_ratio,
                C=C,
                max_iter=10000,
                class_weight='balanced'
            )
            logistic_reg.fit(X_train, y_train)

            # Calculate weighted average AUC across all datasets
            weighted_avg_auc = 0
            aucs = []
            for vDataset in datasets:
                X_val = data.loc[vDataset].drop('amyloid', axis=1)[dataset_features[dataset]]
                y_val = data.loc[vDataset]['amyloid']
                y_pred = logistic_reg.predict(X_val)
                auc = roc_auc_score(y_val, y_pred)
                aucs.append(auc)
                weighted_avg_auc += weights[vDataset] * auc

            if weighted_avg_auc > best_avg_auc:
                best_avg_auc = weighted_avg_auc
                best_model = logistic_reg
                best_params = {'model': 'LR', 'l1_ratio': l1_ratio, 'C': C}

        # Store the best model's AUC for each test dataset
        for vDataset, auc in zip(datasets, aucs):
            roc_auc_results.loc[vDataset, dataset] = auc

    # SVM with L2 Regularization
    for C in tqdm(C_values):
        svm = SVC(kernel='linear', C=C, class_weight='balanced', probability=True)
        svm.fit(X_train, y_train)

        weighted_avg_auc = 0
        aucs = []
        for vDataset in datasets:
            X_val = data.loc[vDataset].drop('amyloid', axis=1)[dataset_features[dataset]]
            y_val = data.loc[vDataset]['amyloid']
            y_pred = svm.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            aucs.append(auc)
            weighted_avg_auc += weights[vDataset] * auc

        if weighted_avg_auc > best_avg_auc:
            best_avg_auc = weighted_avg_auc
            best_model = svm
            best_params = {'model': 'SVM', 'C': C}

    # Store the best model's AUC for each test dataset
    for vDataset, auc in zip(datasets, aucs):
        roc_auc_results.loc[vDataset, dataset] = auc

    # LDA (Linear Discriminant Analysis)
    lda = LDA()
    lda.fit(X_train, y_train)

    weighted_avg_auc = 0
    aucs = []
    for vDataset in datasets:
        X_val = data.loc[vDataset].drop('amyloid', axis=1)[dataset_features[dataset]]
        y_val = data.loc[vDataset]['amyloid']
        y_pred = lda.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        weighted_avg_auc += weights[vDataset] * auc

    if weighted_avg_auc > best_avg_auc:
        best_avg_auc = weighted_avg_auc
        best_model = lda
        best_params = {'model': 'LDA'}

    # Store the best model's AUC for each test dataset
    for vDataset, auc in zip(datasets, aucs):
        roc_auc_results.loc[vDataset, dataset] = auc

    print(f"Top model for dataset {dataset}: {best_params}")

#%%
# visualize heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(roc_auc_results, annot=True, fmt=".2f", cmap="coolwarm_r", vmin=0.5, vmax=1.0)
plt.title('ROC-AUC Scores for Models Trained on Each Dataset')
plt.xlabel('Test Dataset')
plt.ylabel('Train Dataset')
plt.show()














#%%
# visualize the results
# Assuming best_aucs_list is now correctly populated as a 5x5 list of AUC scores
auc_matrix = pd.DataFrame(best_aucs_list, index=datasets, columns=datasets)

plt.figure(figsize=(8, 6))
sns.heatmap(auc_matrix, annot=True, fmt=".2f", cmap="coolwarm_r", vmin=0.5, vmax=1.0)
plt.title('ROC-AUC Scores for Models Trained on Each Dataset')
plt.xlabel('Test Dataset')
plt.ylabel('Train Dataset')
plt.show()

#%%
# Save each best model as a .pkl file
for i, model in enumerate(best_models):
    dataset_name = datasets[i]
    filename = f'best_model_{dataset_name}.pkl'
    joblib.dump(model, filename)
    print(f"Saved {dataset_name} model to {filename}")


