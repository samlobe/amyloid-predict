#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Define data loading functions

def load_data(
    dataset, esm_model, tau_amyloid_def="either", WALTZ_amyloid_def="Classification"
):
    embeddings = pd.read_csv(
        f"../features/{dataset}_{esm_model}_embeddings.csv", index_col=0
    )
    if dataset == "tau":
        y = pd.read_csv("../labels/tau_labels.csv", index_col=0)[tau_amyloid_def]
    elif dataset == "WALTZtht":  # only the WALTZ data with Th-T Binding values
        y = pd.read_csv("../labels/WALTZtht_labels.csv", index_col=0)[WALTZ_amyloid_def]
        y = y.map({"amyloid": True, "non-amyloid": False})
    elif dataset == "TANGO_Table2":
        y = pd.read_csv(
            "../labels/TANGO_Table2_labels.csv", index_col=0
        )["Experimental Aggregation Behavior"]
        y = y.map({"+": True, "-": False})
    # Rename y-column to 'amyloid'
    y = y.rename("amyloid")

    # Merge embeddings and labels
    data = pd.concat([y, embeddings], axis=1)
    return data


def load_all_data(
    esm_model, tau_amyloid_def="either", WALTZ_amyloid_def="Classification"
):
    tau_data = load_data("tau", esm_model, tau_amyloid_def, WALTZ_amyloid_def)
    WALTZtht_data = load_data(
        "WALTZtht", esm_model, tau_amyloid_def, WALTZ_amyloid_def
    )
    TANGO_Table2_data = load_data("TANGO_Table2", esm_model)
    # Merge datasets while labeling the source
    tau_data["dataset"] = "tau"
    WALTZtht_data["dataset"] = "WALTZtht"
    TANGO_Table2_data["dataset"] = "TANGO_Table2"
    data = pd.concat([tau_data, WALTZtht_data, TANGO_Table2_data])
    # Set index to 'dataset' and 'index'
    data = data.set_index(["dataset", data.index])
    return data

#%% Load your data

esm_model = "3B"  # Options: '3B' or '15B'
tau_amyloid_def = "either"  # Options: 'ThT fluorescence', 'pFTAA fluorescence', or 'either'
WALTZ_amyloid_def = "Classification"  # Options: 'Classification' or 'Th-T Binding'

# Select just the tau data
data = load_all_data(esm_model, tau_amyloid_def, WALTZ_amyloid_def).loc["tau"]
y = data["amyloid"]  # 'True' = forms amyloids
X = data.drop("amyloid", axis=1)

#%% Perform LOOCV with Logistic Regression

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
# try SVC
from sklearn.svm import SVC

loo = LeaveOneOut()
n_splits = loo.get_n_splits(X)

y_real = []
y_proba = []

print("Starting LOOCV...")
for train_index, test_index in tqdm(loo.split(X), total=n_splits, desc="LOOCV"):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale the features based on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Logistic Regression model
    lr = LogisticRegression(max_iter=10000, random_state=42,
                            penalty='l2', solver='saga', C=1) # standard L2 penalty
    lr.fit(X_train_scaled, y_train)

    # Get the predicted probabilities for the test sample
    y_test_proba = lr.predict_proba(X_test_scaled)[:, 1]

    # Try SVC quickly
    # svc = SVC(probability=True, random_state=42)
    # svc.fit(X_train_scaled, y_train)
    # y_test_proba = svc.predict_proba(X_test_scaled)[:, 1]


    # Store the true label and predicted probability
    y_real.append(y_test.values[0])
    y_proba.append(y_test_proba[0])

#%% Compute and display evaluation metrics

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

y_real = np.array(y_real)
y_proba = np.array(y_proba)

roc_auc = roc_auc_score(y_real, y_proba)
pr_auc = average_precision_score(y_real, y_proba)

print(f"\nOverall ROC AUC: {roc_auc:.4f}")
print(f"Overall PR AUC: {pr_auc:.4f}")

#%% Plot the ROC curve

fpr, tpr, _ = roc_curve(y_real, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for a random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate",fontsize=14)
plt.ylabel("True Positive Rate",fontsize=14)
plt.title("Receiver Operating Characteristic (ROC) Curve",fontsize=16)
plt.legend(loc="lower right",fontsize=14)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.show()

#%% Plot the Precision-Recall curve

precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall",fontsize=14)
plt.ylabel("Precision",fontsize=14)
plt.title("Precision-Recall Curve",fontsize=16)
plt.legend(loc="lower left",fontsize=14)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.show()
#%%
# # append model, roc_auc, pr_auc to a txt file
# with open("ESM2_15aa_results.txt", "a") as f:
#     f.write(f"esm2-3B, {roc_auc:.4f}, {pr_auc:.4f}\n")
#     f.close()

#%% Train the final model on the full dataset

# Scale the features based on the entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the LR model on the entire dataset
final_model = LogisticRegression(max_iter=10000, random_state=42,
                                 penalty='l2', solver='saga', C=1) # standard L2 penalty
final_model.fit(X_scaled, y)

# Save the final model and scaler
import joblib
joblib.dump({
    'model': final_model,
    'scaler': scaler
}, '../models/15aa_model_latest.joblib')

print("Final model trained and saved as '15aa_model_latest.joblib'.")
#%%