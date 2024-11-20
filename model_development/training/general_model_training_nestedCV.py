#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import joblib

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
# Load your data
esm_model = '3B'  # '3B' or '15B'
tau_amyloid_def = 'either'  # 'ThT fluorescence', 'pFTAA fluorescence', or 'either'
WALTZ_amyloid_def = 'Classification'  # 'Classification' vs 'Th-T Binding'

# Load all datasets
data = load_all_data(esm_model, tau_amyloid_def, WALTZ_amyloid_def)

#%%
# Create 'peptide_length' column based on the dataset
data['peptide_length'] = data.index.get_level_values('dataset').map({
    'tau': 15,
    'WALTZtht': 6,
    'TANGO_Table2': 10
})

#%%
# Create 'stratify_label' for dataset and class combinations
data['stratify_label'] = data.apply(
    lambda row: f"{row['peptide_length']}_{row['amyloid']}", axis=1
)

#%%
# Prepare features and labels
X = data.drop(['amyloid', 'peptide_length', 'stratify_label'], axis=1)
y = data['amyloid']
stratify = data['stratify_label']

#%%
# Set the weighting scheme: 'dataset' or 'dataset_class'
weighting_scheme = 'dataset_class'  # Change to 'dataset_class' for equal dataset+class weighting

#%%
# Calculate sample weights based on the selected weighting scheme
if weighting_scheme == 'dataset':
    # Calculate sample weights based on datasets only
    # Get counts of samples in each dataset
    dataset_counts = data.index.get_level_values('dataset').value_counts()
    num_datasets = dataset_counts.shape[0]
    total_samples = len(data)
    # Calculate the weight for each dataset
    dataset_weights = total_samples / (num_datasets * dataset_counts)
    # Map the sample weights to each sample based on dataset
    data['sample_weight'] = data.index.get_level_values('dataset').map(dataset_weights)
elif weighting_scheme == 'dataset_class':
    # Calculate sample weights based on dataset and class combinations
    category_counts = data['stratify_label'].value_counts()
    num_categories = category_counts.shape[0]
    total_samples = len(data)
    # Calculate the weight for each category
    category_weights = total_samples / (num_categories * category_counts)
    # Map the sample weights to each sample
    data['sample_weight'] = data['stratify_label'].map(category_weights)
else:
    raise ValueError("Invalid weighting_scheme. Choose 'dataset' or 'dataset_class'.")

#%%
# Extract sample weights and dataset names
sample_weight = data['sample_weight']
dataset_names = data.index.get_level_values('dataset')

#%%
# Convert to numpy arrays for efficient indexing
X_array = X.values
y_array = y.values
sample_weight_array = sample_weight.values
dataset_array = dataset_names.values
stratify_array = stratify.values

# Initialize lists to store results
predicted_probs = []
true_labels = []
datasets = []
Cs_used = []

# Initialize LOOCV
loo = LeaveOneOut()

C_values = [1, 0.1, 0.01]
print(f'C values to try: {C_values}')

# Function to select the best C using inner 3-fold stratified CV
def select_best_C(X_train, y_train, sw_train, stratify_train, C_values):
    best_C = None
    best_score = -np.inf  # We aim to maximize the score

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for C in C_values:
        cv_scores = []

        for inner_train_index, inner_valid_index in skf.split(X_train, stratify_train):
            X_inner_train, X_inner_valid = X_train[inner_train_index], X_train[inner_valid_index]
            y_inner_train, y_inner_valid = y_train[inner_train_index], y_train[inner_valid_index]
            sw_inner_train = sw_train[inner_train_index]
            # Stratify labels for inner training set (not needed for fit, but ensuring consistency)
            stratify_inner_train = stratify_train[inner_train_index]

            # Feature scaling
            scaler = StandardScaler()
            scaler.fit(X_inner_train)
            X_inner_train_scaled = scaler.transform(X_inner_train)
            X_inner_valid_scaled = scaler.transform(X_inner_valid)

            # Train model
            model = LogisticRegression(random_state=42, penalty='l2', C=C, max_iter=10000)
            model.fit(X_inner_train_scaled, y_inner_train, sample_weight=sw_inner_train)

            # Predict probabilities on validation set
            y_inner_valid_prob = model.predict_proba(X_inner_valid_scaled)[:, 1]

            # Get score on validation set
            val_score = average_precision_score(y_inner_valid, y_inner_valid_prob)
            cv_scores.append(val_score)

        # Compute average score over the 3 folds
        avg_score = np.mean(cv_scores)

        if avg_score > best_score:
            best_score = avg_score
            best_C = C

    return best_C

# Loop over each sample
for train_index, test_index in tqdm(loo.split(X_array), total=len(X_array), desc="LOO Progress"):
    # Split the data
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]
    sw_train = sample_weight_array[train_index]
    stratify_train = stratify_array[train_index]
    dataset_test = dataset_array[test_index][0]  # Only one sample in test

    # Select the best C using inner 3-fold stratified CV
    best_C = select_best_C(X_train, y_train, sw_train, stratify_train, C_values)

    # Scale the features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model with best C
    model = LogisticRegression(random_state=42, penalty='l2', C=best_C, max_iter=10000)
    model.fit(X_train_scaled, y_train, sample_weight=sw_train)

    # Predict the probability for the test sample
    y_prob = model.predict_proba(X_test_scaled)[0, 1]  # Probability for class 'True'

    # Store the results
    predicted_probs.append(y_prob)
    true_labels.append(y_test[0])
    datasets.append(dataset_test)
    Cs_used.append(best_C)

# Convert lists to numpy arrays
predicted_probs = np.array(predicted_probs)
true_labels = np.array(true_labels)
datasets = np.array(datasets)
Cs_used = np.array(Cs_used)

# Compute ROC curve and ROC AUC
fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
roc_auc = roc_auc_score(true_labels, predicted_probs)

# Compute Precision-Recall curve and PR AUC
precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
pr_auc = average_precision_score(true_labels, predicted_probs)

#%%
# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Overall Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Overall Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

#%%
# Optionally, store predictions in a DataFrame
results_df = pd.DataFrame({
    'predicted_prob': predicted_probs,
    'true_label': true_labels,
    'dataset': datasets,
    'best_C': Cs_used
}, index=data.index)

# Display the first few results
print(results_df.head())

# Initialize dictionaries to store ROC and PR curve data for each dataset
roc_curves = {}
pr_curves = {}

# Unique datasets
unique_datasets = np.unique(datasets)

# Loop over each dataset and compute ROC and PR curves
for dataset in unique_datasets:
    # Filter predictions and true labels by dataset
    dataset_mask = datasets == dataset
    dataset_true_labels = true_labels[dataset_mask]
    dataset_predicted_probs = predicted_probs[dataset_mask]

    # Compute ROC curve and ROC AUC for this dataset
    fpr, tpr, _ = roc_curve(dataset_true_labels, dataset_predicted_probs)
    roc_auc = roc_auc_score(dataset_true_labels, dataset_predicted_probs)
    roc_curves[dataset] = (fpr, tpr, roc_auc)

    # Compute Precision-Recall curve and PR AUC for this dataset
    precision, recall, _ = precision_recall_curve(dataset_true_labels, dataset_predicted_probs)
    pr_auc = average_precision_score(dataset_true_labels, dataset_predicted_probs)
    pr_curves[dataset] = (precision, recall, pr_auc)

#%%

# Plot ROC Curves for all datasets
plt.figure()
fpr_6aa, tpr_6aa, roc_auc_6aa = roc_curves['WALTZtht']
fpr_10aa, tpr_10aa, roc_auc_10aa = roc_curves['TANGO_Table2']
fpr_15aa, tpr_15aa, roc_auc_15aa = roc_curves['tau']
plt.plot(fpr_6aa, tpr_6aa, label=f'6aa (AUC = {roc_auc_6aa:.2f})')
plt.plot(fpr_10aa, tpr_10aa, label=f'10aa (AUC = {roc_auc_10aa:.2f})')
plt.plot(fpr_15aa, tpr_15aa, label=f'15aa (AUC = {roc_auc_15aa:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic by Dataset', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.show()

# Plot Precision-Recall Curves for all datasets
plt.figure()
precision_6aa, recall_6aa, pr_auc_6aa = pr_curves['WALTZtht']
precision_10aa, recall_10aa, pr_auc_10aa = pr_curves['TANGO_Table2']
precision_15aa, recall_15aa, pr_auc_15aa = pr_curves['tau']
plt.plot(recall_6aa, precision_6aa, label=f'6aa (avg precision = {pr_auc_6aa:.2f})')
plt.plot(recall_10aa, precision_10aa, label=f'10aa (avg precision = {pr_auc_10aa:.2f})')
plt.plot(recall_15aa, precision_15aa, label=f'15aa (avg precision = {pr_auc_15aa:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve by Dataset', fontsize=16)
plt.legend(loc='lower left', fontsize=13)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.show()

# Optionally, store predictions in a DataFrame
results_df = pd.DataFrame({
    'predicted_prob': predicted_probs,
    'true_label': true_labels,
    'dataset': datasets,
    'best_C': Cs_used
}, index=data.index)

# Display the first few results
print(results_df.head())

#%% 
# turn the Cs_used into strings
results_df['best_C'] = results_df['best_C'].astype(str)

# histogram of best C values
plt.figure()
results_df['best_C'].value_counts().plot(kind='bar')
plt.xlabel('Best C Value')
plt.ylabel('Frequency')

#%%

# Determine the most frequently selected C value from LOOCV
# Since Cs_used might be strings, ensure they are converted to floats
Cs_used_float = [float(c) for c in Cs_used]
C_counter = Counter(Cs_used_float)
best_C = C_counter.most_common(1)[0][0]

print(f"The most frequently selected C value is: {best_C}")

#%%
# Scale the full dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

# Train the final logistic regression model using the best C
final_model = LogisticRegression(random_state=42, penalty='l2', C=best_C, max_iter=10000)
final_model.fit(X_scaled, y_array, sample_weight=sample_weight_array)

# Save the trained model and scaler using joblib
joblib.dump({
    'model': final_model,
    'scaler': scaler
}, '../models/general_model_latest.joblib')

print("Final model trained and saved as 'general_model_final_latest.joblib'.")

#%%