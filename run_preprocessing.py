import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_valid_techniques():
    with open('techniques_subtask3.txt', 'r') as f:
        valid_techniques = [line.strip() for line in f.readlines()]
    return set(valid_techniques)

def filter_techniques(techniques_str, valid_techniques):
    if pd.isna(techniques_str):
        return ''
    
    techniques = techniques_str.split(',')
    valid = [t.strip() for t in techniques if t.strip() in valid_techniques]
    return ','.join(valid)

def preprocess_final_data():
    os.makedirs('_processed_data', exist_ok=True)
    
    valid_techniques = load_valid_techniques()
    
    print("Loading datasets...")
    current_data = pd.read_csv('_input_data/...', sep='\t')
    old_data = pd.read_csv('_input_data/...', sep='\t')
    
    lang_to_keep = ['ru', 'po']
    old_data = old_data[old_data['language'].isin(lang_to_keep)]
    
    print("Splitting current data into train and test sets...")
    train_data, test_data = train_test_split(current_data, test_size=300, random_state=42, stratify=current_data['language'])
    
    print("Filtering techniques in old data...")
    old_data['techniques'] = old_data['techniques'].apply(
        lambda x: filter_techniques(x, valid_techniques)
    )
    
    print("Combining datasets...")
    train_data = pd.concat([train_data, old_data], ignore_index=True)
    
    print("Creating one-hot encodings...")
    for technique in valid_techniques:
        # For training data
        train_data[technique] = train_data['techniques'].apply(
            lambda x: 1 if technique in str(x).split(',') else 0
        )
        # For test data
        test_data[technique] = test_data['techniques'].apply(
            lambda x: 1 if technique in str(x).split(',') else 0
        )
    
    print("Saving processed datasets...")
    
    train_path = '_processed_data/...'
    train_data.to_csv(train_path, index=False)
    print(f"train path: {train_path}")
    
    test_path = '_processed_data/...'
    test_data.to_csv(test_path, index=False)
    print(f"test path: {test_path}")

    print("\nProcessing completed!")
    
    print("\nLanguage distribution in training set:")
    for lang in list(train_data['language'].unique()):
        count = len(train_data[train_data['language'] == lang])
        percentage = (count / len(train_data)) * 100
        print(f"{lang}: {count} samples ({percentage:.2f}%)")

    print("\nLanguage distribution in testing set:")
    for lang in list(test_data['language'].unique()):
        count = len(test_data[test_data['language'] == lang])
        percentage = (count / len(test_data)) * 100
        print(f"{lang}: {count} samples ({percentage:.2f}%)")
    
if __name__ == "__main__":
    preprocess_final_data()