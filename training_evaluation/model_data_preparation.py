from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, valid_techniques=None, has_targets=True):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.segment_text
        self.valid_techniques = valid_techniques
        self.has_targets = has_targets
        self.max_len = max_len
        
        if has_targets and valid_techniques is not None:
            self.targets = dataframe[valid_techniques].values
        else:
            self.targets = None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        
        item = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        }
        
        if self.has_targets and self.targets is not None:
            item['targets'] = torch.tensor(self.targets[index], dtype=torch.float)
            
        return item

def load_valid_techniques():
    with open('techniques_subtask3.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

def create_dataset_training(MODEL_NAME: str, train_path: str, test_path: str, TRAINING_LANGUAGES: list[str] = None):
    """
    Function to load and process the data into suitable format.
    
    Args:
        MODEL_NAME: name of the model as per Hugging face to use for tokenizer
        train_path: path to training CSV file
        test_path: path to test CSV file
        TRAINING_LANGUAGES: List of languages to include. If None, use all languages.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, return_dict=False)
    MAX_LEN = min(tokenizer.model_max_length, 512)
    print(f'max length of tokens for < {MODEL_NAME} > is: < {MAX_LEN} >')

    valid_techniques = load_valid_techniques()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TRAINING_LANGUAGES is not None:
        train_df = train_df[train_df['language'].isin(TRAINING_LANGUAGES)]
        test_df = test_df[test_df['language'].isin(TRAINING_LANGUAGES)]

    print(f"TRAIN Dataset: {train_df.shape}")
    print(f"TEST Dataset: {test_df.shape}")
    print(f"Languages in training data: {train_df['language'].unique()}")
    print(f"Languages in test data: {test_df['language'].unique()}")

    # Create datasets
    train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, valid_techniques, has_targets=True)
    test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN, valid_techniques, has_targets=True)

    return train_dataset, test_dataset

def create_dataloader(args, test_df, language, valid_techniques=None, has_targets=True):
    """Create a dataloader for a specific language"""
    filtered_test_df = test_df[test_df['language'] == language].reset_index(drop=True)
    
    tokenizer, MAX_LEN = get_tokenizer(args['MODEL_NAME'])
    test_dataset = CustomDataset(filtered_test_df, tokenizer, MAX_LEN, valid_techniques, has_targets=has_targets)

    if args['ONLY_TEST']:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args['BATCH_SIZE'],
            shuffle=args['SHUFFLE'],
            num_workers=args['NUM_WORKERS'],
            valid_techniques=None,
            has_targets=False
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args['BATCH_SIZE'],
            shuffle=args['SHUFFLE'],
            num_workers=args['NUM_WORKERS'],
        )
    
    return test_loader, filtered_test_df

def get_tokenizer(model_name):
    """Get tokenizer and max length for a given model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)
    MAX_LEN = min(tokenizer.model_max_length, 512)
    return tokenizer, MAX_LEN