#%%
import matplotlib.pyplot as plt
import pandas as pd

# Load performance metrics for WALTZall
performance_metrics_all = pd.read_csv('WALTZall_featNum_performance.csv')
# Load performance metrics for WALTZtht
performance_metrics_tht = pd.read_csv('WALTZtht_featNum_performance.csv')

# Plotting
plt.figure(figsize=(8, 5))

# WALTZtht - darker colors, circles
plt.plot(performance_metrics_tht['num_features_used'], performance_metrics_tht['roc_auc'], 
         label='WALTZ (ThT subset): ROC AUC', marker='o', color='tab:blue')
plt.plot(performance_metrics_tht['num_features_used'], performance_metrics_tht['pr_auc'], 
         label='WALTZ (ThT subset): avg precision', marker='o', color='tab:orange')

# WALTZall - lighter colors, squares
plt.plot(performance_metrics_all['num_features_used'], performance_metrics_all['roc_auc'], 
         label='WALTZ (all data): ROC AUC', marker='s', color='tab:blue', alpha=0.5, markerfacecolor='none')
plt.plot(performance_metrics_all['num_features_used'], performance_metrics_all['pr_auc'], 
         label='WALTZ (all data): avg precision', marker='s', color='tab:orange', alpha=0.5, markerfacecolor='none')

# Adding labels, title, and legend
plt.xlabel('features in model', fontsize=14)
plt.ylabel('performance', fontsize=14)
#plt.title('Performance vs Number of Features Used', fontsize=16)
plt.ylim([0.5, 1])
plt.xlim([0.5,25])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='lower right',fontsize=14)
plt.grid(True)
plt.show()
