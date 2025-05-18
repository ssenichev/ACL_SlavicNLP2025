import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_valid_techniques():
    with open('techniques_subtask3.txt', 'r') as f:
        valid_techniques = [line.strip() for line in f.readlines()]
    return set(valid_techniques)

def has_any_technique(techniques_str, valid_techniques):
    """
    Returns 1 if any valid technique is present, 0 otherwise.
    """
    if pd.isna(techniques_str) or techniques_str.strip() == '':
        return 0
    techniques = [t.strip() for t in techniques_str.split(',')]
    for t in techniques:
        if t in valid_techniques:
            return 1
    return 0

def preprocess_binary_data():
    os.makedirs('_processed_data', exist_ok=True)
    
    valid_techniques = load_valid_techniques()
    
    print("Loading datasets...")
    current_data = pd.read_csv('_input_data/...', sep='\t')
    old_data = pd.read_csv('_input_data/...', sep='\t')

    lang_to_keep = ['en', 'ru', 'po']
    old_data = old_data[old_data['language'].isin(lang_to_keep)]
    
    print("Splitting current data into train and test sets...")
    train_data, test_data = train_test_split(current_data, test_size=200, random_state=42, stratify=current_data['language'])
    
    print("Assigning binary labels in old data...")
    old_data['has_technique'] = old_data['techniques'].apply(
        lambda x: has_any_technique(x, valid_techniques)
    )
    
    print("Combining datasets...")
    train_data = pd.concat([train_data, old_data], ignore_index=True)
    
    print("Assigning binary labels to train and test sets...")
    train_data['has_technique'] = train_data['techniques'].apply(
        lambda x: has_any_technique(x, valid_techniques)
    )
    test_data['has_technique'] = test_data['techniques'].apply(
        lambda x: has_any_technique(x, valid_techniques)
    )
    
    print("Saving processed datasets...")
    train_data.to_csv('_processed_data/...', index=False)
    test_data.to_csv('_processed_data/...', index=False)
    
    print("\nProcessing completed!")
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    print("\nLabel distribution in training set:")
    for label in [0, 1]:
        count = (train_data['has_technique'] == label).sum()
        percentage = (count / len(train_data)) * 100
        print(f"has_technique={label}: {count} samples ({percentage:.2f}%)")

if __name__ == "__main__":
    preprocess_binary_data()
