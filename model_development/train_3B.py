#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tqdm import tqdm

#%% import features from esm_3B and labels
def load_data(dataset,esm_model,tau_amyloid_def='either',WALTZ_amyloid_def='Classification'):
    embeddings = pd.read_csv(f'features/{dataset}_{esm_model}_embeddings.csv',index_col=0)
    if dataset == 'tau':
        y = pd.read_csv('labels/tau_labels.csv',index_col=0)[tau_amyloid_def]
    elif dataset == 'WALTZtht': # only the WALTZ data with Th-T Binding values
        y = pd.read_csv('labels/WALTZtht_labels.csv',index_col=0)[WALTZ_amyloid_def]
        y = y.map({'amyloid':True,'non-amyloid':False})
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
    WALTZtht_data = load_data('WALTZtht',esm_model,tau_amyloid_def,WALTZ_amyloid_def)
    TANGO_Table2_data = load_data('TANGO_Table2',esm_model)
    # merge datasets while labeling the source
    tau_data['dataset'] = 'tau'
    WALTZtht_data['dataset'] = 'WALTZtht'
    TANGO_Table2_data['dataset'] = 'TANGO_Table2'
    data = pd.concat([tau_data,WALTZtht_data,TANGO_Table2_data])
    # set index to 'dataset' and 'index'
    data = data.set_index(['dataset',data.index])
    return data

#%%
esm_model = '3B' # 3B or 15B
tau_amyloid_def = 'either' # ThT fluorescence or pFTAA fluorescene or either (at least 50% more fluorescence than bulk)
WALTZ_amyloid_def = 'Classification' # "Classification" vs "Th-T Binding"

data = load_all_data(esm_model,tau_amyloid_def,WALTZ_amyloid_def)

# Directory to store the models
model_save_dir = 'models_3B'  # You can change this to your preferred directory
# Make sure the directory exists
os.makedirs(model_save_dir, exist_ok=True)


#%%

# for each dataset, try 10 values of C and note the auc scores when testing on the other 3 datasets
# finally plot avg auc score vs C for each dataset

datasets = ['tau', 'WALTZtht', 'TANGO_Table2']
C_values = np.logspace(0, 1.5, 30)
best_C_values = {}
best_feature_sets = {}
aucs_lists = []

for dataset in datasets:
    X_train = data.loc[dataset].drop('amyloid', axis=1)
    y_train = data.loc[dataset]['amyloid']

    best_C = None
    aucs = []
    features_lists = []

    for C in tqdm(C_values):
        model = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', random_state=69, C=C)
        model.fit(X_train, y_train)

        for test_dataset in datasets:
            X_test = data.loc[test_dataset].drop('amyloid', axis=1)
            y_test = data.loc[test_dataset]['amyloid']
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            aucs.append(auc)

            # what are the nonzero features?
            features = X_train.columns[model.coef_[0] != 0]
            features_lists.append(features)
    aucs_lists.append(aucs)
    mean_aucs = np.mean(np.array(aucs).reshape(-1, len(datasets)), axis=1)
    # plot C values vs ROC-AUC
    plt.plot(C_values, mean_aucs, label=dataset)
    plt.xlabel('C')
    plt.ylabel('ROC-AUC')
    plt.title(f'Training on {dataset} dataset')
    plt.show()

    # find the best C
    best_C_index = np.argmax(mean_aucs)
    best_C = C_values[best_C_index]
    best_C_values[dataset] = float(best_C)
    best_feature_sets[dataset] = features_lists[best_C_index]

# print the best features from each dataset
for dataset, features in best_feature_sets.items():
    # print the number of features
    print(f'Number of features for {dataset} dataset: {len(features)}')
    # print(f"Best features for {dataset} dataset: {features}")

# print the best C values for each dataset
for dataset, best_C in best_C_values.items():
    print(f"Best C for {dataset} dataset: {best_C}")


#%%
# get the set of these selected features
selected_features = set()
for features in best_feature_sets.values():
    selected_features.update(features)

selected_features = list(selected_features)

# print the number of selected features
print(f"Number of selected features: {len(selected_features)}")

# order these based on the embedding index
selected_features = sorted(selected_features, key=lambda x: int(x.split('_')[-1]))

# save a csv file with the selected features
pd.Series(selected_features).to_csv(f'{model_save_dir}/selected_features_{esm_model}.csv', index=False, header=False)


#%%


# Initialize parameters
C_values_lr = np.logspace(-2, 2, 30)
l1_ratios = [0, 0.25, 0.5]
C_values_svc = np.logspace(-2, 1, 30)
models = {}
roc_auc_results = pd.DataFrame(index=datasets, columns=datasets)

# Initialize lists to store AUC values for plotting
lr_auc_vs_c = {l1_ratio: [] for l1_ratio in l1_ratios}
svc_auc_vs_c = []

# Train models using selected features
for train_dataset in datasets:
    X_train = data.loc[train_dataset].drop('amyloid', axis=1)[selected_features]
    y_train = data.loc[train_dataset]['amyloid']

    best_model = None
    best_avg_auc = -np.inf
    best_params = None
    best_aucs_for_heatmap = {}

    # Logistic Regression with Elastic Net
    for l1_ratio in l1_ratios:
        auc_values = []
        for C in tqdm(C_values_lr):
            logistic_reg = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=l1_ratio,
                C=C,
                max_iter=10000,
                class_weight='balanced',
                random_state=69
            )
            logistic_reg.fit(X_train, y_train)

            # Validate on other datasets
            aucs = {}
            for test_dataset in datasets:
                X_test = data.loc[test_dataset].drop('amyloid', axis=1)[selected_features]
                y_test = data.loc[test_dataset]['amyloid']
                y_pred = logistic_reg.predict_proba(X_test)[:,1]
                auc = roc_auc_score(y_test, y_pred)
                aucs[test_dataset] = auc
            avg_auc = np.mean(list(aucs.values()))
            auc_values.append(avg_auc)  # Store the average AUC for this C

            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_model = logistic_reg
                best_params = {'model': 'LR', 'l1_ratio': l1_ratio, 'C': C}
                best_aucs_for_heatmap = aucs  # Store AUCs for heatmap

        lr_auc_vs_c[l1_ratio] = auc_values  # Store all AUC values for this l1_ratio

    # Plot C vs avg AUC for Logistic Regression (Elastic Net)
    plt.figure(figsize=(8, 6))
    for l1_ratio in l1_ratios:
        plt.plot(C_values_lr, lr_auc_vs_c[l1_ratio], label=f'l1_ratio={l1_ratio}')
    plt.xscale('log')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Average ROC-AUC')
    plt.title(f'C vs Average ROC-AUC for Logistic Regression (Train: {train_dataset})')
    plt.legend()
    plt.show()

    # SVM with L2 Regularization
    auc_values = []
    for C in tqdm(C_values_svc):
        svm = SVC(kernel='linear', C=C, probability=True, class_weight='balanced', random_state=69)
        svm.fit(X_train, y_train)

        aucs = {}
        for test_dataset in datasets:
            X_test = data.loc[test_dataset].drop('amyloid', axis=1)[selected_features]
            y_test = data.loc[test_dataset]['amyloid']
            y_pred = svm.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_pred)
            aucs[test_dataset] = auc
        avg_auc = np.mean(list(aucs.values()))
        auc_values.append(avg_auc)  # Store the average AUC for this C

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_model = svm
            best_params = {'model': 'SVM', 'C': C}
            best_aucs_for_heatmap = aucs  # Store AUCs for heatmap

    svc_auc_vs_c = auc_values  # Store all AUC values for SVM

    # Plot C vs avg AUC for SVM
    plt.figure(figsize=(8, 6))
    plt.plot(C_values_svc, svc_auc_vs_c, label='SVM')
    plt.xscale('log')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Average ROC-AUC')
    plt.title(f'C vs Average ROC-AUC for SVM (Train: {train_dataset})')
    plt.legend()
    plt.show()

    # LDA (Linear Discriminant Analysis)
    lda = LDA()
    lda.fit(X_train, y_train)

    aucs = {}
    for test_dataset in datasets:
        X_test = data.loc[test_dataset].drop('amyloid', axis=1)[selected_features]
        y_test = data.loc[test_dataset]['amyloid']
        y_pred = lda.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_pred)
        aucs[test_dataset] = auc
    avg_auc = np.mean(list(aucs.values()))

    if avg_auc > best_avg_auc:
        best_avg_auc = avg_auc
        best_model = lda
        best_params = {'model': 'LDA'}
        best_aucs_for_heatmap = aucs  # Store AUCs for heatmap

    # Store the best model's AUC for each test dataset
    for test_dataset, auc in best_aucs_for_heatmap.items():
        roc_auc_results.loc[test_dataset, train_dataset] = auc

    models[train_dataset] = best_model
    print(f"Top model for training on {train_dataset}: {best_params}")

# Generate heatmap
roc_auc_array = roc_auc_results.astype(float).to_numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(roc_auc_array, annot=True, fmt=".3f", cmap="coolwarm_r", linewidths=0.5, vmin=0.5, vmax=1)
plt.title('ROC-AUC heatmap for top models\ntrained on each dataset', fontsize=16)
plt.xlabel('Train Datasets', fontsize=14)
plt.ylabel('Test Datasets', fontsize=14)
cbar = plt.gca().collections[0].colorbar
cbar.set_label('ROC-AUC',fontsize=14)
cbar.ax.tick_params(labelsize=14)
for t in plt.gca().texts:
    t.set_fontsize(14)

plt.xticks(ticks=np.arange(len(datasets))+0.5, labels=datasets, rotation=0, fontsize=14)
plt.yticks(ticks=np.arange(len(datasets))+0.5, labels=datasets, rotation=0, fontsize=14)
plt.show()

#%%
# save the models
# Save each model after training
for train_dataset, model in models.items():
    model_filename = f'{model_save_dir}/{train_dataset}_top_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model for {train_dataset} saved to {model_filename}")

#%%
# Ensemble the 3 models into one final model!
# Parameters
n_samples_per_dataset = 1000  # Adjust based on your needs

# Initialize lists to store the new features and labels
logit_features = []
y_ensemble = []

# Loop over each dataset
for dataset in datasets:
    X = data.loc[dataset].drop('amyloid', axis=1)[selected_features]
    y = data.loc[dataset]['amyloid']
    
    # Extract logits for each model
    tau_logits = models['tau'].decision_function(X)
    waltz_logits = models['WALTZtht'].decision_function(X)
    tango_logits =  models['TANGO_Table2'].decision_function(X)
    
    # Stack logits together to form new features
    logits = np.vstack([tau_logits, waltz_logits, tango_logits]).T  # Shape: (n_samples, 3)
    
    # Balance the classes within this dataset
    X_resampled, y_resampled = resample(logits, y, 
                                        replace=True, 
                                        n_samples=n_samples_per_dataset,
                                        stratify=y, 
                                        random_state=69)
    
    logit_features.append(X_resampled)
    y_ensemble.append(y_resampled)

# Combine all datasets to create the final training data for the ensemble model
X_ensemble = np.vstack(logit_features)
y_ensemble = np.concatenate(y_ensemble)

# Train the ensemble model with class weights balanced
ensemble_model = LogisticRegression(max_iter=1000, class_weight='balanced',random_state=69)
# ensemble_model = SVC(kernel='linear', class_weight='balanced', probability=True)
ensemble_model.fit(X_ensemble, y_ensemble)

print("Ensemble model trained successfully.")

# To save the ensemble model
joblib.dump(ensemble_model, f'{model_save_dir}/ensemble_model.joblib')

#%% apply the ensembled model to each dataset and get confusion matrix and ROC for each dataset
confusion_matrices = {}

# plot the ROC curve for the ensemble model on each dataset
plt.figure(figsize=(8, 6))
for dataset in datasets:
    X = data.loc[dataset].drop('amyloid', axis=1)[selected_features]
    y = data.loc[dataset]['amyloid']
    
    tau_logits = models['tau'].decision_function(X)
    waltz_logits = models['WALTZtht'].decision_function(X)
    tango_logits = models['TANGO_Table2'].decision_function(X)
    
    logits = np.vstack([tau_logits, waltz_logits, tango_logits]).T  # Shape: (n_samples, 3)
    
    y_pred_label = ensemble_model.predict(logits)
    confusion_matrices[dataset] = confusion_matrix(y, y_pred_label)

    y_pred_proba = ensemble_model.predict_proba(logits)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{dataset} dataset')

    # Print the ROC-AUC score for the ensemble model
    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f"ROC-AUC of ensembled model on {dataset} dataset: {roc_auc:.3f}")

# Define class names for labeling the confusion matrix
class_names = ['non-amyloid', 'amyloid']  # Assuming binary classification

# Print the confusion matrices in a prettier way
for dataset, cm in confusion_matrices.items():
    print(f"\nConfusion matrix for {dataset} dataset:")
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(df_cm)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Ensembled Model', fontsize=16)
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)

#%%
# in case you want to see results of the 3 sub-models
# # Initialize an empty dictionary to store confusion matrices for the tau-trained model predictions
# confusion_matrices_oneModel = {}
# which_model = 'WALTZtht'

# # Loop over each dataset to generate predictions using the tau-trained model
# for dataset in datasets:
#     # Prepare the data for the current dataset
#     X = data.loc[dataset].drop('amyloid', axis=1)[selected_features]
#     y = data.loc[dataset]['amyloid']
    
#     # apply the tau-trained model
#     y_pred = models[which_model].predict(X)

#     # Generate the confusion matrix
#     confusion_matrices_oneModel[dataset] = confusion_matrix(y, y_pred)

# # Print the confusion matrices in a prettier way
# for dataset, cm in confusion_matrices_oneModel.items():
#     print(f"\nConfusion matrix for {dataset} dataset ({which_model}-trained model):")
#     df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
#     print(df_cm)