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
import joblib
from tqdm import tqdm

# Import features from esm_3B and labels
def load_data(dataset, esm_model, tau_amyloid_def='either', WALTZ_amyloid_def='Classification'):
    embeddings = pd.read_csv(f'../features/{dataset}_{esm_model}_embeddings.csv', index_col=0)
    if dataset == 'tau':
        y = pd.read_csv('../labels/tau_labels.csv', index_col=0)[tau_amyloid_def]
    elif dataset == 'WALTZtht':  # only the WALTZ data with Th-T Binding values
        y = pd.read_csv('../labels/WALTZtht_labels.csv', index_col=0)[WALTZ_amyloid_def]
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
    WALTZtht_data = load_data('WALTZtht', esm_model, tau_amyloid_def, WALTZ_amyloid_def)
    TANGO_Table2_data = load_data('TANGO_Table2', esm_model)
    # merge datasets while labeling the source
    tau_data['dataset'] = 'tau'
    WALTZtht_data['dataset'] = 'WALTZtht'
    TANGO_Table2_data['dataset'] = 'TANGO_Table2'
    data = pd.concat([tau_data, WALTZtht_data, TANGO_Table2_data])
    # set index to 'dataset' and 'index'
    data = data.set_index(['dataset', data.index])
    return data

#%%
# Load your data as before
esm_model = '3B'  # 3B or 15B
tau_amyloid_def = 'either'  # ThT fluorescence or pFTAA fluorescence or either
WALTZ_amyloid_def = 'Classification'  # "Classification" vs "Th-T Binding"

# Select just the tau data
data = load_all_data(esm_model, tau_amyloid_def, WALTZ_amyloid_def).loc['WALTZtht']
y = data['amyloid']  # 'True' = forms amyloids
X = data.drop('amyloid', axis=1)

#%% Split the data into training and validation sets
# Ensure stratification to maintain class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

#%% Find an appropriate range of L1 Cs that produce 1-12 features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

#%%
# List to store the number of features selected for each C
num_features_selected = []
C_values = np.linspace(0.01, 0.17, 35)  # Spanning the desired range of up to 12 features

for C in tqdm(C_values):
    # Fit logistic regression with L1 penalty
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=10000,
        C=C,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    
    # Count the number of non-zero coefficients
    num_nonzero = np.sum(model.coef_ != 0)
    num_features_selected.append(num_nonzero)

# Plot the number of features selected vs. C values
plt.figure(figsize=(8, 6))
plt.plot(C_values, num_features_selected, marker='o')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Number of Features Selected')
plt.title('L1 Regularization: Number of Features Selected vs. C')
plt.grid(True)
plt.show()

#%% Define the list of regularization strengths for L1 and L2 penalties
C_list_L1 = np.linspace(0.01, 0.17, 35) # when considering several possible selections of features
C_list_L1 = [0.076] # after zeroing in on the 10-features model (see CV_6aa_2step_featNum.py)
C_list_L2 = [0.01, 0.05, 0.1, 0.5, 1, 5]
print(f'L1 Cs: {C_list_L1}')
print(f'L2 Cs: {C_list_L2}')

#%% Perform hyperparameter tuning on the training set
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize variables to store the best hyperparameters
best_score = -np.inf
best_params = {'features': None, 'C': None}

# Scale features based on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Perform feature selection using L1 regularization on the entire training set
feature_sets = []
print("Performing feature selection with L1 regularization...")
for C_l1 in C_list_L1:
    # Fit logistic regression with L1 penalty
    l1_model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=10000,
        C=C_l1,
        random_state=42,
    )
    l1_model.fit(X_train_scaled, y_train)

    # Get indices of non-zero coefficients
    nonzero_indices = np.where(l1_model.coef_[0] != 0)[0]
    feature_set = tuple(nonzero_indices)
    feature_sets.append(feature_set)

print(f"Unique feature sets obtained: {len(feature_sets)}")

# Inner cross-validation for hyperparameter tuning
print("Evaluating models with selected features and L2 regularization...")
for feature_set in feature_sets:
    X_train_fs = X_train_scaled[:, feature_set]

    # Inner cross-validation for L2 regularization strength
    for C_l2 in C_list_L2:
        inner_scores = []

        # Inner cross-validation loop
        for inner_train_index, inner_val_index in inner_cv.split(X_train_fs, y_train):
            X_train_inner = X_train_fs[inner_train_index]
            y_train_inner = y_train.iloc[inner_train_index]
            X_val_inner = X_train_fs[inner_val_index]
            y_val_inner = y_train.iloc[inner_val_index]

            # Check if inner training data contains both classes
            if len(np.unique(y_train_inner)) < 2:
                # Skip this inner fold
                continue

            # Train logistic regression model with L2 penalty
            model = LogisticRegression(
                penalty='l2',
                solver='saga',
                max_iter=10000,
                C=C_l2,
                random_state=42,
            )
            model.fit(X_train_inner, y_train_inner)

            # Predict probabilities on the validation set
            y_val_pred_proba = model.predict_proba(X_val_inner)[:, 1]
            # Measure validation score with f1 score
            # val_score = average_precision_score(y_val_inner, y_val_pred_proba)
            val_score = f1_score(y_val_inner, y_val_pred_proba > 0.5)
            inner_scores.append(val_score)

        # Compute the mean validation score for the current hyperparameters
        if len(inner_scores) > 0:
            mean_inner_score = np.mean(inner_scores)
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params['features'] = feature_set
                best_params['C'] = C_l2

# After hyperparameter tuning
if best_params['features'] is None:
    print("No valid hyperparameters found.")
    exit()

# Train logistic regression model with the best hyperparameters on the entire training set
X_train_fs = X_train_scaled[:, best_params['features']]
X_val_fs = X_val_scaled[:, best_params['features']]

final_model = LogisticRegression(
    penalty='l2',
    solver='saga',
    max_iter=10000,
    C=best_params['C'],
    random_state=42,
)
final_model.fit(X_train_fs, y_train)

# Predict probabilities on the validation set
y_val_pred_proba = final_model.predict_proba(X_val_fs)[:, 1]

#%% 
# print the number of features selected and the best hyperparameters
print(f"Number of Features Selected: {len(best_params['features'])}")
print(f"Best C: {best_params['C']:.4f}")

# Evaluate the model on the validation set
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

roc_auc = roc_auc_score(y_val, y_val_pred_proba)
pr_auc = average_precision_score(y_val, y_val_pred_proba)

print(f"\nValidation ROC AUC: {roc_auc:.4f}")
print(f"Validation PR AUC: {pr_auc:.4f}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for a random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('6aa ROC Curve on Validation Set:\n10-feature model', fontsize=16)
plt.xticks(fontsize=14) ; plt.yticks(fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.show()

# Plot the Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
plt.figure()
plt.plot(recall, precision, label=f'PR curve (Average Precision = {pr_auc:.2f})')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.xticks(fontsize=14) ; plt.yticks(fontsize=14)
plt.title('6aa Precision-Recall Curve on Validation Set:\n10-feature model', fontsize=16)
plt.legend(loc="lower left", fontsize=14)
plt.show()

#%% Train Final Model! (not excluding 30% for validation)

# ReLoad the full dataset
data_full = load_all_data(esm_model, tau_amyloid_def, WALTZ_amyloid_def).loc['WALTZtht']
y_full = data_full['amyloid']  # 'True' = forms amyloids
X_full = data_full.drop('amyloid', axis=1)
# standardize the full dataset
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Try many Cs to find which one selects 10 features
num_features_selected = []
C_values = np.linspace(0.01, 0.06, 25)  # Spanning the desired range of up to 12 features

for C in tqdm(C_values):
    # Fit logistic regression with L1 penalty
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=10000,
        C=C,
        random_state=42,
    )
    model.fit(X_full_scaled, y_full)
    
    # Count the number of non-zero coefficients
    num_nonzero = np.sum(model.coef_ != 0)
    num_features_selected.append(num_nonzero)

# Plot the number of features selected vs. C values
plt.figure(figsize=(8, 6))
plt.plot(C_values, num_features_selected, marker='o')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Number of Features Selected')
plt.title('L1 Regularization: Number of Features Selected vs. C')
plt.grid(True)
plt.show()

#%% Pull out the C-value that selects 10 features
C_list_L1 = 0.039 # after zeroing in on the 10-features model (didn't change when considering full dataset)
C_list_L2 = [0.01, 0.05, 0.1, 0.5, 1, 5]
print(f'L1 Cs: {C_list_L1}')
print(f'L2 Cs: {C_list_L2}')

# train on X_full_scaled
best_score = -np.inf
best_params = {'features': None, 'C': None}

# Perform feature selection using L1 regularization on the entire training set
print("Performing feature selection with L1 regularization...")
# Fit logistic regression with L1 penalty
l1_model = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=10000,
    C=C_list_L1,
    random_state=42,
)
l1_model.fit(X_full_scaled, y_full)

# Get indices of non-zero coefficients
nonzero_indices = np.where(l1_model.coef_[0] != 0)[0]
feature_set = tuple(nonzero_indices)

#%% Perform hyperparameter tuning on the full dataset
X_full_selected = X_full_scaled[:, feature_set]

best_params = {'C': None}
best_score = -np.inf
# Inner cross-validation for L2 regularization strength
for C_l2 in C_list_L2:
    inner_scores = []

    # Inner cross-validation loop
    for inner_train_index, inner_val_index in inner_cv.split(X_full_selected, y_full):
        X_train_inner = X_full_selected[inner_train_index]
        y_train_inner = y_full.iloc[inner_train_index]
        X_val_inner = X_full_selected[inner_val_index]
        y_val_inner = y_full.iloc[inner_val_index]
        
        # Train logistic regression model with L2 penalty
        model = LogisticRegression(
            penalty='l2',
            solver='saga',
            max_iter=10000,
            C=C_l2,
            random_state=42,
        )
        model.fit(X_train_inner, y_train_inner)

        # Predict probabilities on the validation set
        y_val_pred_proba = model.predict_proba(X_val_inner)[:, 1]
        # Measure validation score with f1 score
        val_score = f1_score(y_val_inner, y_val_pred_proba > 0.5)
        inner_scores.append(val_score)

    # Compute the mean validation score for the current hyperparameters
    if len(inner_scores) > 0:
        mean_inner_score = np.mean(inner_scores)
        if mean_inner_score > best_score:
            best_score = mean_inner_score
            best_params['C'] = C_l2

FETA_model = LogisticRegression(
    penalty='l2',
    solver='saga',
    max_iter=10000,
    C=best_params['C'],
    random_state=42,
)
FETA_model.fit(X_full_selected, y_full)

print(f"Number of Features Selected: {len(feature_set)}")
print(f"Best C: {best_params['C']:.4f}")

# Save the FETA model
joblib.dump({
    'model': FETA_model,
    'scaler': scaler,
    'feature_set': feature_set,
    'C': best_params['C'],
}, '../models/6aa_FETA_model_latest.joblib'
)

print("FETA model trained and saved as '6aa_FETA_model_latest.joblib'.")

feature_set = [int(i) for i in feature_set]
print("Feature set (0-indexed):")
print(feature_set)

#%%
### now retrain 24-feature model
# (marginally better performance but less interpretable than FETA)
num_features_selected = []
C_values = np.linspace(0.06, 0.10, 3)  # Spanning the desired range of up to 12 features

for C in tqdm(C_values):
    # Fit logistic regression with L1 penalty
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=10000,
        C=C,
        random_state=42,
    )
    model.fit(X_full_scaled, y_full)
    
    # Count the number of non-zero coefficients
    num_nonzero = np.sum(model.coef_ != 0)
    num_features_selected.append(num_nonzero)

# Plot the number of features selected vs. C values
plt.figure(figsize=(8, 6))
plt.plot(C_values, num_features_selected, marker='o')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Number of Features Selected')
plt.title('L1 Regularization: Number of Features Selected vs. C')
plt.grid(True)
plt.show()

#%% Pull out the C-value that selects 24 features
C_list_L1 = 0.08 # after zeroing in on the 24-features model
C_list_L2 = [0.01, 0.05, 0.1, 0.5, 1, 5]
print(f'L1 Cs: {C_list_L1}')
print(f'L2 Cs: {C_list_L2}')

# train on X_full_scaled
best_score = -np.inf
best_params = {'features': None, 'C': None}

# Perform feature selection using L1 regularization on the entire training set
print("Performing feature selection with L1 regularization...")
# Fit logistic regression with L1 penalty
l1_model = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=10000,
    C=C_list_L1,
    random_state=42,
)
l1_model.fit(X_full_scaled, y_full)

# Get indices of non-zero coefficients
nonzero_indices = np.where(l1_model.coef_[0] != 0)[0]
feature_set = tuple(nonzero_indices)

#%% Perform hyperparameter tuning on the full dataset
X_full_selected = X_full_scaled[:, feature_set]

best_params = {'C': None}
best_score = -np.inf
# Inner cross-validation for L2 regularization strength
for C_l2 in C_list_L2:
    inner_scores = []

    # Inner cross-validation loop
    for inner_train_index, inner_val_index in inner_cv.split(X_full_selected, y_full):
        X_train_inner = X_full_selected[inner_train_index]
        y_train_inner = y_full.iloc[inner_train_index]
        X_val_inner = X_full_selected[inner_val_index]
        y_val_inner = y_full.iloc[inner_val_index]
        
        # Train logistic regression model with L2 penalty
        model = LogisticRegression(
            penalty='l2',
            solver='saga',
            max_iter=10000,
            C=C_l2,
            random_state=42,
        )
        model.fit(X_train_inner, y_train_inner)

        # Predict probabilities on the validation set
        y_val_pred_proba = model.predict_proba(X_val_inner)[:, 1]
        # Measure validation score with f1 score
        val_score = f1_score(y_val_inner, y_val_pred_proba > 0.5)
        inner_scores.append(val_score)

    # Compute the mean validation score for the current hyperparameters
    if len(inner_scores) > 0:
        mean_inner_score = np.mean(inner_scores)
        if mean_inner_score > best_score:
            best_score = mean_inner_score
            best_params['C'] = C_l2

#%%

best_model = LogisticRegression(
    penalty='l2',
    solver='saga',
    max_iter=10000,
    C=best_params['C'],
    random_state=42,
)
best_model.fit(X_full_selected, y_full)

print(f"Number of Features Selected: {len(feature_set)}")
print(f"Best C: {best_params['C']:.4f}")

# Save this best model
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'feature_set': feature_set,
    'C': best_params['C'],
}, '../models/6aa_best_model_latest.joblib'
)

print("Best model (24-features) trained and saved as '6aa_best_model_latest.joblib'.")

