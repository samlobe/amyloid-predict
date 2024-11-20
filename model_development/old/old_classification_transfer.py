#%%
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# load embeddings
esm_model = '3B' # 3B or 15B
type = 'tau' # tau or WALTZ or TANGO_Table1 or TANGO_Table2
tau_amyloid_def = 'either' # ThT or pFTAA or either (at least 50% more fluorescence than bulk)
WALTZ_amyloid_def = 'Classification' # "Classification" vs "Th-T Binding"
def load_data(esm_model,type,tau_amyloid_def='either',WALTZ_amyloid_def='Classification'):
    embeddings = pd.read_csv(f'{type}_{esm_model}_embeddings.csv',index_col=0)
    if type == 'tau':
        y = pd.read_csv('tau_labels.csv',index_col=0)[tau_amyloid_def]
    elif type == 'WALTZ':
        y = pd.read_csv('WALTZ_labels.csv',index_col=0)[WALTZ_amyloid_def]
        y = y.map({'amyloid':True,'non-amyloid':False})
    elif type == 'TANGO_Table1':
        y = pd.read_csv('TANGO_Table1_labels.csv',index_col=0)['Experimental Aggregation Behavior']
        y = y.map({'+':True,'-':False})
    elif type == 'TANGO_Table2':
        y = pd.read_csv('TANGO_Table2_labels.csv',index_col=0)['Experimental Aggregation Behavior']
        y = y.map({'+':True,'-':False})
    # rename y-column to 'amyloid'
    y = y.rename('amyloid')

    # merge embeddings and labels
    data = pd.concat([y,embeddings],axis=1)
    return data

def load_all_data(esm_model,tau_amyloid_def='either',WALTZ_amyloid_def='Classification'):
    tau_data = load_data(esm_model,'tau',tau_amyloid_def,WALTZ_amyloid_def)
    WALTZ_data = load_data(esm_model,'WALTZ',tau_amyloid_def,WALTZ_amyloid_def)
    TANGO_Table1_data = load_data(esm_model,'TANGO_Table1')
    TANGO_Table2_data = load_data(esm_model,'TANGO_Table2')
    # merge tau and WALTZ data while labeling the source
    tau_data['source'] = 'tau'
    WALTZ_data['source'] = 'WALTZ'
    TANGO_Table1_data['source'] = 'TANGO_Table1'
    TANGO_Table2_data['source'] = 'TANGO_Table2'
    data = pd.concat([tau_data,WALTZ_data,TANGO_Table1_data,TANGO_Table2_data])
    # set index to 'source' and 'index'
    data = data.set_index(['source',data.index])
    return data

model_types = {
    'lda': LinearDiscriminantAnalysis(),
    'lr': LogisticRegression(max_iter=1000),
    'lr_l1': LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', C=0.1),
    'svm': SVC(probability=True)
}

def train_model(data, model_type, C=0.1):
    model = model_types[model_type]

    X = data.drop('amyloid',axis=1)
    y = data['amyloid']
    model.fit(X,y)
    return model

# data = load_data(esm_model,type,tau_amyloid_def,WALTZ_amyloid_def)
all_data = load_all_data(esm_model,tau_amyloid_def,WALTZ_amyloid_def)
tau_data = all_data.loc['tau']
WALTZ_data = all_data.loc['WALTZ']
TANGO_Table1_data = all_data.loc['TANGO_Table1']
TANGO_Table2_data = all_data.loc['TANGO_Table2']

#%%
# Define the datasets and models
datasets = ['tau', 'WALTZ', 'TANGO_Table1', 'TANGO_Table2']

# Initialize a dictionary to store AUC-ROC scores
roc_auc_scores = {model: {train: {test: None for test in datasets} for train in datasets} for model in model_types}

# Training and evaluation
for train_ds in datasets:
    train_data = all_data.loc[train_ds]
    X_train = train_data.drop('amyloid', axis=1)
    y_train = train_data['amyloid']
    
    for model_name, model in model_types.items():
        # Train the model on the training dataset
        trained_model = train_model(train_data, model_name)
        
        for test_ds in datasets:
            test_data = all_data.loc[test_ds]
            X_test = test_data.drop('amyloid', axis=1)
            y_test = test_data['amyloid']
            
            # Evaluate the model on the test dataset
            y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            roc_auc_scores[model_name][train_ds][test_ds] = roc_auc

# Now roc_auc_scores contains all the AUC-ROC scores for all combinations of train-test datasets and models

#%%
import seaborn as sns

# For each model, create a heatmap of the ROC-AUC scores
for model_name, scores_matrix in roc_auc_scores.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(scores_matrix), annot=True, fmt=".2f", cmap="coolwarm_r", vmin=0.5, vmax=1.0)
    plt.title(f'ROC-AUC Scores for {model_name}')
    plt.xlabel('Train Dataset')
    plt.ylabel('Test Dataset')
    plt.show()

#%%
# Define the range of C values to test
C_values = np.logspace(0.8, 1.3, 20)  # from 0.001 to 10

# Initialize a dictionary to store AUC-ROC scores for each C value
roc_auc_scores_C = {C: {train: {test: None for test in datasets} for train in datasets} for C in C_values}

# Loop over C values
for C in C_values:
    # Update the model type dictionary for logistic regression with the current C value
    model_types['lr_l1'] = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', C=C)
    
    # Train and evaluate the model for each dataset as training and testing set
    for train_ds in datasets:
        train_data = all_data.loc[train_ds]
        X_train = train_data.drop('amyloid', axis=1)
        y_train = train_data['amyloid']
        
        trained_model = train_model(train_data, 'lr_l1', C=C)
        
        for test_ds in datasets:
            test_data = all_data.loc[test_ds]
            X_test = test_data.drop('amyloid', axis=1)
            y_test = test_data['amyloid']
            
            # Evaluate the model
            y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            roc_auc_scores_C[C][train_ds][test_ds] = roc_auc

# Plotting the results for each C value
for C, scores_matrix in roc_auc_scores_C.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(scores_matrix), annot=True, fmt=".2f", cmap="coolwarm_r", vmin=0.5, vmax=1.0)
    plt.title(f'ROC-AUC Scores for Logistic Regression with L1 penalty (C={C:.3f})')
    plt.xlabel('Test Dataset')
    plt.ylabel('Train Dataset')
    plt.show()

#%% try feature selection with LR L1
# Set the value of C for feature selection
C = 11

# Load Tau data
tau_data = all_data.loc['tau']
X_tau = tau_data.drop('amyloid', axis=1)
y_tau = tau_data['amyloid']

# Train Logistic Regression with L1 penalty to perform feature selection
model_l1_tau = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', C=C)
model_l1_tau.fit(X_tau, y_tau)

# Identify features with non-zero coefficients
nonzero_features_tau = X_tau.columns[model_l1_tau.coef_[0] != 0]
print(f"Selected features from Tau dataset: {nonzero_features_tau}")

#%%
C = 2
# Load WALTZ data
WALTZ_data = all_data.loc['WALTZ']
X_W = WALTZ_data.drop('amyloid', axis=1)
y_W = WALTZ_data['amyloid']

# Train Logistic Regression with L1 penalty to perform feature selection
model_l1_W = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', C=C)
model_l1_W.fit(X_W, y_W)

# Identify features with non-zero coefficients
nonzero_features_W = X_W.columns[model_l1_W.coef_[0] != 0]
print(f"Selected features from WALTZ dataset: {nonzero_features_W}")

#%%
# nonzero features from either set
selected_features = set(nonzero_features_tau) | set(nonzero_features_W)


#%%
# Load all data as before
all_data = load_all_data(esm_model)

# Use the selected features from the Tau dataset across all models
# selected_features = nonzero_features_tau  # Already computed in your code
# selected_features = nonzero_features_W
selected_features = selected_features

# Initialize a dictionary to store AUC-ROC scores
roc_auc_scores = {model: {train: {test: None for test in datasets} for train in datasets} for model in model_types}

# Training and evaluation using selected features
for train_ds in datasets:
    train_data = all_data.loc[train_ds]
    X_train = train_data[selected_features]  # Use only selected features
    y_train = train_data['amyloid']
    
    for model_name, model in model_types.items():
        # Train the model on the training dataset using selected features
        model.fit(X_train, y_train)
        
        for test_ds in datasets:
            test_data = all_data.loc[test_ds]
            X_test = test_data[selected_features]  # Evaluate using the same selected features
            y_test = test_data['amyloid']
            
            # Evaluate the model on the test dataset
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            roc_auc_scores[model_name][train_ds][test_ds] = roc_auc

# Visualizing the AUC-ROC scores as before
import seaborn as sns

for model_name, scores_matrix in roc_auc_scores.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(scores_matrix), annot=True, fmt=".2f", cmap="coolwarm_r", vmin=0.5, vmax=1.0)
    plt.title(f'ROC-AUC Scores for {model_name}')
    plt.xlabel('Train Dataset')
    plt.ylabel('Test Dataset')
    plt.show()
