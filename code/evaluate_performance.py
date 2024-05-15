import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

ADDITIONAL_FEATURES = True

file = f'counterspeech_results_additional_features_{ADDITIONAL_FEATURES}'

roc_df = pd.read_csv(f'../data/{file}.csv')
            
print(f"ROC-AUC: {roc_df['roc_auc'].mean()} +/- {roc_df['roc_auc'].std() / np.sqrt(len(roc_df))}")

true_positives = pd.read_csv(f'../data/{file}_true_positives.csv')
false_positives = pd.read_csv(f'../data/{file}_false_positives.csv')
true_negatives = pd.read_csv(f'../data/{file}_true_negatives.csv')
false_negatives = pd.read_csv(f'../data/{file}_false_negatives.csv')

threshold_columns = true_positives.filter(regex='threshold').columns

thresholds = np.arange(0, 1, 0.05)
f1_threshold_means = []
f1_threshold_errors = []
accuracy_threshold_means = []
accuracy_threshold_errors = []
precision_threshold_means = []
precision_threshold_errors = []
recall_threshold_means = []
recall_threshold_errors = []
for i, col in enumerate(threshold_columns): #go through each possible threshold
    f1_scores = []
    precisions = []
    recalls = []
    accuracies = []
    for seed in range(50): #go through 
        if seed not in true_positives.index:
            print(file)
            display(false_positives)
            
        TP = true_positives.loc[seed, col]
        FP = false_positives.loc[seed, col]
        TN = true_negatives.loc[seed, col]
        FN = false_negatives.loc[seed, col]

        precision = TP/(TP+FP) if TP+FP > 0 else 0 # Of all "class X" calls I made, how many were right?
        recall = TP/(TP+FN) if TP+FN > 0 else 0 # Of all "class X" calls I should have made, how many did I actually make?
        f1 = (2*precision*recall)/(precision+recall) if precision+recall > 0 else 0
        accuracy = (TP+TN)/(TP+FP+TN+FN)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)
    
    f1_threshold_means.append(np.mean(f1_scores))
    f1_threshold_errors.append(np.std(f1_scores) / np.sqrt(len(f1_scores)))
    accuracy_threshold_means.append(np.mean(accuracies))
    accuracy_threshold_errors.append(np.std(accuracies) / np.sqrt(len(accuracies)))
    precision_threshold_means.append(np.mean(precisions))
    precision_threshold_errors.append(np.std(precisions) / np.sqrt(len(precisions)))
    
    recall_threshold_means.append(np.mean(recalls))
    recall_threshold_errors.append(np.std(recalls) / np.sqrt(len(recalls)))

    
print(f"F1: {np.max(f1_threshold_means)} +/- {f1_threshold_errors[np.argmax(f1_threshold_means)]}")
    
print(f"Precision: {precision_threshold_means[np.argmax(f1_threshold_means)]} +/- {precision_threshold_errors[np.argmax(f1_threshold_means)]}")

print(f"Recall: {recall_threshold_means[np.argmax(f1_threshold_means)]} +/- {recall_threshold_errors[np.argmax(f1_threshold_means)]}")

    