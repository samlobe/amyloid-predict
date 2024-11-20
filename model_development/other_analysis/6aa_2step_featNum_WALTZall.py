#%%

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import features from esm_3B and labels
def load_data(dataset, esm_model, tau_amyloid_def='either', WALTZ_amyloid_def='Classification'):
    embeddings = pd.read_csv(f'../features/{dataset}_{esm_model}_embeddings.csv', index_col=0)
    if dataset == 'tau':
        y = pd.read_csv('../labels/tau_labels.csv', index_col=0)[tau_amyloid_def]
    elif dataset == 'WALTZtht':  # only the WALTZ data with Th-T Binding values
        y = pd.read_csv('../labels/WALTZtht_labels.csv', index_col=0)[WALTZ_amyloid_def]
        y = y.map({'amyloid': True, 'non-amyloid': False})
    elif dataset == 'WALTZall':
        y = pd.read_csv('../labels/WALTZall_labels.csv', index_col=0)[WALTZ_amyloid_def]
        y = y.map({'amyloid': True, 'non-amyloid': False})
    elif dataset == 'TANGO_Table2':
        y = pd.read_csv('../labels/TANGO_Table2_labels.csv', index_col=0)['Experimental Aggregation Behavior']
        y = y.map({'+': True, '-': False})
    # rename y-column to 'amyloid'
    y = y.rename('amyloid')

    # merge embeddings and labels
    data = pd.concat([y, embeddings], axis=1)
    return data

def load_all_data(esm_model, tau_amyloid_def='either', WALTZ_amyloid_def='Classification'):
    tau_data = load_data('tau', esm_model, tau_amyloid_def, WALTZ_amyloid_def)
    WALTZall_data = load_data('WALTZall', esm_model, tau_amyloid_def, WALTZ_amyloid_def)
    TANGO_Table2_data = load_data('TANGO_Table2', esm_model)
    # merge datasets while labeling the source
    tau_data['dataset'] = 'tau'
    WALTZall_data['dataset'] = 'WALTZall'
    TANGO_Table2_data['dataset'] = 'TANGO_Table2'
    data = pd.concat([tau_data, WALTZall_data, TANGO_Table2_data])
    # set index to 'dataset' and 'index'
    data = data.set_index(['dataset', data.index])
    return data

#%%
# Load the data
esm_model = '3B'  # 3B or 15B
tau_amyloid_def = 'either'  # ThT fluorescence or pFTAA fluorescence or either
WALTZ_amyloid_def = 'Classification'  # "Classification" vs "Th-T Binding"

# Select just the WALTZall data
data = load_all_data(esm_model, tau_amyloid_def, WALTZ_amyloid_def).loc['WALTZall']
y = data['amyloid']  # 'True' = forms amyloids
X = data.drop('amyloid', axis=1)

# select just the WALTZtht data
data_tht = load_data('WALTZtht',esm_model, tau_amyloid_def, WALTZ_amyloid_def)
y_tht = data_tht['amyloid']  # 'True' = forms amyloids
X_tht = data_tht.drop('amyloid', axis=1)

_, X_val_tht, _, y_val_tht = train_test_split(
    X_tht, y_tht, test_size=0.3, stratify=y_tht, random_state=42
)

val_indices = X_val_tht.index
train_data = data.loc[~data.index.isin(val_indices)]
X_train = train_data.drop('amyloid', axis=1)
y_train = train_data['amyloid']

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Initialize variables to store performance metrics
performance_metrics = {
    'L1_C_values': [],
    'num_features_allowed': [],
    'num_features_used': [],
    'best_L2_C': [],
    'roc_auc': [],
    'pr_auc': []
}

# List to store unique feature sets
evaluated_feature_sets = []

# Define a range of L1 C values to explore for feature selection
L1_C_values = np.linspace(0.0043, 0.025, 120)  # Adjust the range and number of points as needed
# L1_C_values = np.linspace(0.01, 0.03, 3) # finding upper limit

# Define a range of L2 C values to explore for model training
L2_C_values = [0.01, 0.05, 0.1, 0.5, 1, 5]

# Scale features based on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val_tht) # test on subset of ThT data

print("Evaluating models with increasing L1 C values for feature selection and tuning L2 C values...")

# Loop over the L1 C values
for L1_C in tqdm(L1_C_values, desc='Processing L1 C values'):
    # Fit logistic regression with L1 penalty to select features
    l1_model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=10000,
        C=L1_C,
        random_state=42,
        n_jobs=-1
    )
    l1_model.fit(X_train_scaled, y_train)

    # Get indices of non-zero coefficients (selected features)
    nonzero_indices = np.where(l1_model.coef_[0] != 0)[0]
    num_features_selected = len(nonzero_indices)

    # Create a tuple of selected features to check uniqueness
    feature_set = tuple(nonzero_indices)

    # Only proceed if we have not already evaluated this feature set
    if feature_set not in evaluated_feature_sets and num_features_selected > 0:
        evaluated_feature_sets.append(feature_set)

        # Prepare data with selected features
        X_train_fs = X_train_scaled[:, nonzero_indices]
        X_val_fs = X_val_scaled[:, nonzero_indices]

        # Inner cross-validation to select best L2 C value
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for L2_C in L2_C_values:
            fold_scores = []
            for train_idx, val_idx in inner_cv.split(X_train_fs, y_train):
                X_train_cv, X_val_cv = X_train_fs[train_idx], X_train_fs[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train logistic regression model with L2 penalty on selected features
                l2_model_cv = LogisticRegression(
                    penalty='l2',
                    solver='saga',
                    max_iter=10000,
                    C=L2_C,
                    random_state=42,
                    n_jobs=-1
                )
                l2_model_cv.fit(X_train_cv, y_train_cv)

                # Predict probabilities on the validation fold
                y_val_pred_proba_cv = l2_model_cv.predict_proba(X_val_cv)[:, 1]

                # Evaluate performance (using f1 score)
                # pr_auc_cv = average_precision_score(y_val_cv, y_val_pred_proba_cv)
                pr_auc_cv = f1_score(y_val_cv, y_val_pred_proba_cv > 0.5)
                fold_scores.append(pr_auc_cv)

            # Compute mean CV score for this L2 C value
            mean_cv_score = np.mean(fold_scores)
            cv_scores.append(mean_cv_score)

        # Select the L2 C value with the best average CV score
        best_idx = np.argmax(cv_scores)
        best_L2_C = L2_C_values[best_idx]

        # Retrain L2 logistic regression model on entire training data with selected features and best L2 C
        final_l2_model = LogisticRegression(
            penalty='l2',
            solver='saga',
            max_iter=10000,
            C=best_L2_C,
            random_state=42,
            n_jobs=-1
        )
        final_l2_model.fit(X_train_fs, y_train)

        # Predict probabilities on the validation set
        y_val_pred_proba = final_l2_model.predict_proba(X_val_fs)[:, 1]

        # Evaluate performance
        roc_auc = roc_auc_score(y_val_tht, y_val_pred_proba) # evaluating on 
        pr_auc = average_precision_score(y_val_tht, y_val_pred_proba)

        # Record the performance metrics
        performance_metrics['L1_C_values'].append(L1_C)
        performance_metrics['num_features_allowed'].append(num_features_selected)  # Since L1 C controls features indirectly
        performance_metrics['num_features_used'].append(num_features_selected)
        performance_metrics['best_L2_C'].append(best_L2_C)
        performance_metrics['roc_auc'].append(roc_auc)
        performance_metrics['pr_auc'].append(pr_auc)

        print(f"L1 C: {L1_C:.4f}, Num Features: {num_features_selected}, Best L2 C: {best_L2_C}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

print("\nCompleted evaluation over all L1 C values.")

#%%
# Plotting the performance metrics vs number of features used
plt.figure(figsize=(10,6))
plt.plot(performance_metrics['num_features_used'], performance_metrics['roc_auc'], label='ROC AUC', marker='o')
plt.plot(performance_metrics['num_features_used'], performance_metrics['pr_auc'], label='PR AUC', marker='o')
plt.xlabel('Number of Features Used', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.title('Model Performance vs Number of Features Used', fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
#%%
# note the repeats at 20 and 24; only use the first one
performance_metrics_df = pd.DataFrame(performance_metrics)
performance_metrics_df = performance_metrics_df.drop_duplicates(subset=['num_features_used'])
performance_metrics_df = performance_metrics_df.reset_index(drop=True)

# replot
plt.figure(figsize=(8,5))
plt.plot(performance_metrics_df['num_features_used'], performance_metrics_df['roc_auc'], label='ROC AUC', marker='o')
plt.plot(performance_metrics_df['num_features_used'], performance_metrics_df['pr_auc'], label='Average Precision', marker='o')
plt.xlabel('Number of Features Used', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.title('6aa Amyloid Classifier', fontsize=16)
# plt.xlim([0,25])
plt.ylim([0.5,1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
#%%
performance_metrics_df.to_csv('WALTZall_featNum_performance.csv')