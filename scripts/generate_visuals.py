#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Generates and saves visualizations for the project."""
    
    # Load data
    train_df = pd.read_csv("data/processed/politifact_train.std.csv")
    val_df = pd.read_csv("data/processed/politifact_val.std.csv")
    test_df = pd.read_csv("data/processed/politifact_test.std.csv")
    
    # --- 1. Dataset Overview ---
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Number of rows
    print(f"Total number of rows: {len(all_df)}")
    
    # Number of fake and real news
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=all_df)
    plt.title('Distribution of Fake and Real News')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig('reports/figures/dataset_overview.png')
    print("Saved dataset overview plot.")
    
    # --- 2. Train/Test Split Sizes ---
    split_sizes = {
        'Train': len(train_df),
        'Validation': len(val_df),
        'Test': len(test_df)
    }
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(split_sizes.keys()), y=list(split_sizes.values()))
    plt.title('Train, Validation, and Test Set Sizes')
    plt.xlabel('Set')
    plt.ylabel('Number of Samples')
    plt.savefig('reports/figures/train_test_split.png')
    print("Saved train/test split plot.")

    # --- 3. Model vs. Accuracy Tables (Split into CV and Test) ---
    exp_summary = pd.read_csv("reports/tables/experiments_summary.csv")
    
    # Drop the model_path column
    if 'model_path' in exp_summary.columns:
        exp_summary = exp_summary.drop(columns=['model_path'])
        
    # Shorten scenario names
    scenario_map = {
        'baseline_linear_svc': 'BL_LSVC',
        'cleaned_only_linear_svc': 'CO_LSVC',
        'selected_linear_svc': 'SEL_LSVC',
        'selected_no_engineered_linear_svc': 'SNE_LSVC',
        'baseline_random_forest': 'BL_RF',
        'cleaned_only_random_forest': 'CO_RF',
        'selected_random_forest': 'SEL_RF',
        'selected_no_engineered_random_forest': 'SNE_RF',
        'baseline_xgboost': 'BL_XGB',
        'cleaned_only_xgboost': 'CO_XGB',
        'selected_xgboost': 'SEL_XGB',
        'selected_no_engineered_xgboost': 'SNE_XGB',
    }
    exp_summary['scenario'] = exp_summary['scenario'].replace(scenario_map)

    # Round float columns to 2 decimal places
    for col in exp_summary.columns:
        if exp_summary[col].dtype == 'float64':
            exp_summary[col] = exp_summary[col].round(2)

    # --- Table 1: CV Accuracies ---
    cv_summary = exp_summary[['scenario', 'model', 'cv_acc', 'cv_f1']].copy()
    cv_summary.rename(columns={'cv_acc': 'CV Accuracy', 'cv_f1': 'CV F1-Score'}, inplace=True)
    
    fig1, ax1 = plt.subplots(figsize=(8, 3)) # Adjusted size for fewer columns
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=cv_summary.values, colLabels=cv_summary.columns, cellLoc = 'center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.2)
    plt.title('Scenario vs. CV Accuracy', y=1.2)
    plt.savefig("reports/figures/scenario_vs_cv_accuracy.png", bbox_inches='tight', pad_inches=0.1)
    print("Saved CV accuracy table.")

    # --- Table 2: Test Accuracies ---
    test_summary = exp_summary[['scenario', 'model', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall']].copy()
    test_summary.rename(columns={
        'test_accuracy': 'Test Accuracy', 
        'test_f1': 'Test F1-Score',
        'test_precision': 'Precision',
        'test_recall': 'Recall'
    }, inplace=True)

    fig2, ax2 = plt.subplots(figsize=(10, 3)) # Adjusted size for more columns
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=test_summary.values, colLabels=test_summary.columns, cellLoc = 'center', loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.2)
    plt.title('Scenario vs. Test Accuracy', y=1.2)
    plt.savefig("reports/figures/scenario_vs_test_accuracy.png", bbox_inches='tight', pad_inches=0.1)
    print("Saved test accuracy table.")


if __name__ == "__main__":
    main()
