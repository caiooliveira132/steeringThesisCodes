import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.stats import pearsonr

# Load dataset
df = pd.read_csv('newResults_outdoor_drowsy_augmented.csv')
if 'Filename' in df.columns:
    df = df.drop(columns=['Filename'])
df['KSS_state'] = df['KSS_state'].astype(int)

# Define fixed class labels
labels = sorted(df['KSS_state'].unique())

# Feature definitions
single_features = ['TradSE', 'ApproxSE', 'SampleSE',
                   'Skewness', 'Kurtosis', 'StdDev', 'Peak2Peak']
entropy_features = ['TradSE', 'ApproxSE', 'SampleSE']
base_features = ['Skewness', 'Kurtosis', 'StdDev', 'Peak2Peak']
all_features = base_features + entropy_features
label_col = 'KSS_state'

# Initialize scaler and cross-validator
scaler = StandardScaler()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model constructors
model_constructors = [
    ('LogisticRegression', lambda: LogisticRegression(max_iter=1000)),
    ('GaussianNB', GaussianNB),
    ('DecisionTree', lambda: DecisionTreeClassifier(random_state=42)),
    ('RandomForest', lambda: RandomForestClassifier(random_state=42)),
    ('SVM', lambda: SVC(kernel='rbf', random_state=42))
]

# Evaluation function


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        'ConfusionMatrix': confusion_matrix(y, y_pred, labels=labels),
        'Correlation': pearsonr(y, y_pred)[0],
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1Score': f1_score(y, y_pred, average='weighted')
    }


fold_results = []
summary_results = []

# Single-feature evaluation
for feat in single_features:
    for name, ctor in model_constructors:
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(df[[feat]], df[label_col]), start=1):
            X_train = scaler.fit_transform(df.loc[train_idx, [feat]])
            X_test = scaler.transform(df.loc[test_idx, [feat]])
            y_train = df.loc[train_idx, label_col]
            y_test = df.loc[test_idx, label_col]

            model = ctor()
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            fold_results.append(
                {**metrics, 'Type': 'Single', 'Features': feat, 'Model': name, 'Fold': fold})
            fold_metrics.append(metrics)

        df_f = pd.DataFrame(fold_metrics)
        agg = {
            'Type': 'Single',
            'Features': feat,
            'Model': name,
            'Fold': 'All',
            'ConfusionMatrix': np.rint(np.stack(df_f['ConfusionMatrix'].tolist()).mean(axis=0)).astype(int)
        }
        for m in ['Accuracy', 'Precision', 'Recall', 'F1Score', 'Correlation']:
            agg[f'{m}_Mean'] = df_f[m].mean()
            agg[f'{m}_Std'] = df_f[m].std()
        summary_results.append(agg)

# Base + one entropy feature
for ent in entropy_features:
    combo = base_features + [ent]
    combo_name = '+'.join(combo)
    for name, ctor in model_constructors:
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(df[combo], df[label_col]), start=1):
            X_train = scaler.fit_transform(df.loc[train_idx, combo])
            X_test = scaler.transform(df.loc[test_idx, combo])
            y_train = df.loc[train_idx, label_col]
            y_test = df.loc[test_idx, label_col]

            model = ctor()
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            fold_results.append(
                {**metrics, 'Type': 'Combo', 'Features': combo_name, 'Model': name, 'Fold': fold})
            fold_metrics.append(metrics)

        df_f = pd.DataFrame(fold_metrics)
        agg = {
            'Type': 'Combo',
            'Features': combo_name,
            'Model': name,
            'Fold': 'All',
            'ConfusionMatrix': np.rint(np.stack(df_f['ConfusionMatrix'].tolist()).mean(axis=0)).astype(int)
        }
        for m in ['Accuracy', 'Precision', 'Recall', 'F1Score', 'Correlation']:
            agg[f'{m}_Mean'] = df_f[m].mean()
            agg[f'{m}_Std'] = df_f[m].std()
        summary_results.append(agg)

# All features together
combo_name = '+'.join(all_features)
for name, ctor in model_constructors:
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(df[all_features], df[label_col]), start=1):
        X_train = scaler.fit_transform(df.loc[train_idx, all_features])
        X_test = scaler.transform(df.loc[test_idx, all_features])
        y_train = df.loc[train_idx, label_col]
        y_test = df.loc[test_idx, label_col]

        model = ctor()
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        fold_results.append(
            {**metrics, 'Type': 'All', 'Features': combo_name, 'Model': name, 'Fold': fold})
        fold_metrics.append(metrics)

    df_f = pd.DataFrame(fold_metrics)
    agg = {
        'Type': 'All',
        'Features': combo_name,
        'Model': name,
        'Fold': 'All',
        'ConfusionMatrix': np.rint(np.stack(df_f['ConfusionMatrix'].tolist()).mean(axis=0)).astype(int)
    }
    for m in ['Accuracy', 'Precision', 'Recall', 'F1Score', 'Correlation']:
        agg[f'{m}_Mean'] = df_f[m].mean()
        agg[f'{m}_Std'] = df_f[m].std()
    summary_results.append(agg)

# Save results
pd.DataFrame(fold_results).to_csv(
    'newResults_outdoor_drowsy_augmented_folds.csv', index=False)
pd.DataFrame(summary_results).to_csv(
    'newResults_outdoor_drowsy_augmented_summary.csv', index=False)

print("5-Fold CV completed. Results saved.")
