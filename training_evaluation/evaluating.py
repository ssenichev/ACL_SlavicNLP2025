from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter
from training_evaluation.model_data_preparation import CustomDataset, get_tokenizer, load_valid_techniques
from training_evaluation.models import BERTBase, EnglishRobertaBase, EnglishRobertaLarge, mBERTBase, XLMRobertaBase, XLMRobertaLarge, XLMRobertaL_ParlaMint, mBERT_Slavic

import warnings
warnings.filterwarnings('always')
warnings.simplefilter('always')


def get_model_class(model_name):
    model_classes = {
        'roberta-large': EnglishRobertaLarge,
        'roberta-base': EnglishRobertaBase,
        'bert-base-uncased': BERTBase,
        'bert-base-multilingual-cased': mBERTBase,
        'xlm-roberta-base': XLMRobertaBase,
        'xlm-roberta-large': XLMRobertaLarge,
        'classla/xlm-r-parla': XLMRobertaL_ParlaMint,
        'DeepPavlov/bert-base-bg-cs-pl-ru-cased': mBERT_Slavic
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model_classes[model_name]


def test(model, testing_loader, only_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            if not only_test and 'targets' in data:
                targets = data['targets'].to(device, dtype=torch.float)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())

            outputs = model(ids, mask, token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets if not only_test else None


def run_tests(model, testing_loader, find_threshold, evaluation_threshold, only_test, lang, results_list):
    outputs, targets = test(model, testing_loader, only_test)

    def get_metrics():
        if targets is None:
            return 0.0
        accuracy = metrics.accuracy_score(targets, thresholded_outputs)
        f1_score_micro = metrics.f1_score(targets, thresholded_outputs, average='micro', zero_division=0)
        f1_score_macro = metrics.f1_score(targets, thresholded_outputs, average='macro', zero_division=0)
        results_list.append([lang, current_threshold, accuracy, f1_score_micro, f1_score_macro])
        return f1_score_micro
    
    best_f1_micro = 0
    best_threshold = evaluation_threshold
    best_outputs = None
    
    if find_threshold and not only_test:
        for current_threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            thresholded_outputs = np.array(outputs) >= current_threshold
            current_f1_micro = get_metrics()
            if current_f1_micro > best_f1_micro:
                best_f1_micro = current_f1_micro
                best_threshold = current_threshold
                best_outputs = thresholded_outputs
    else:
        thresholded_outputs = np.array(outputs) >= evaluation_threshold
        if not only_test:
            best_f1_micro = get_metrics()
        best_threshold = evaluation_threshold
        best_outputs = thresholded_outputs

    if not only_test:
        print(f"\nBest threshold for {lang}: {best_threshold} with F1-micro score: {best_f1_micro}")
    return best_outputs, results_list, best_threshold



def create_submission_txt(outputs, df, path, mappings, binary=False):
    result_df = df[['documentID', 'start', 'end']].copy()
    result_df['documentID'] = result_df['documentID'].apply(lambda x: x if x.endswith('.txt') else f"{x}.txt")
    
    rows = []
    for i, output in enumerate(outputs):
        filename = result_df.iloc[i]['documentID']
        start = result_df.iloc[i]['start']
        end = result_df.iloc[i]['end']
        
        techniques = [technique for technique, keep in zip(mappings, output) if keep]
        row = [filename, str(start), str(end)]
    
        for technique in techniques:
            row.append(technique)
        
        rows.append(row)
    
    with open(path, 'w', encoding='utf-8') as f:
        for i, row in enumerate(rows):
            if i < len(rows) - 1:
                f.write('\t'.join(row) + '\n')
            else:
                f.write('\t'.join(row))

def run_eval(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = args["MODEL_FILE_NAME"].split("/")[-1].split(".")[0]
    path_to_folder = f'results/{args["RESULT_DIR_NAME"]}/{model_name}/'
    os.makedirs(path_to_folder, exist_ok=True)
    
    checkpoint = torch.load(args['MODEL_FILE_NAME'])
    print(f"LOADING MODEL FROM {args['MODEL_FILE_NAME']}")
    
    model_class = get_model_class(checkpoint['args']['MODEL_NAME'])
    num_labels = 1 if args['BINARY'] == True else len(load_valid_techniques())
    model = model_class(num_labels=num_labels)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data and valid techniques
    test_df = pd.read_csv(args['EVALUATION_SET'])
    valid_techniques = load_valid_techniques() if not args['BINARY'] else None
    
    language_counts = Counter(test_df.language)
    results_list = []
    
    languages_to_evaluate = args['LANGUAGES'] if args['LANGUAGES'] is not None else list(language_counts.keys())
    
    for language in languages_to_evaluate:
        print(f'\n\n=========== Now evaluating language: {language} ===========\n')

        if language_counts.get(language, 'na') == 'na':
            print('no data points to be tested')
        else:
            lang_df = test_df[test_df['language'] == language].reset_index(drop=True)
            
            tokenizer, max_len = get_tokenizer(args['MODEL_NAME'])
            dataset = CustomDataset(
                lang_df, 
                tokenizer, 
                max_len, 
                valid_techniques=valid_techniques,
                has_targets=not args['ONLY_TEST']
            )
            testing_loader = DataLoader(
                dataset,
                batch_size=args['BATCH_SIZE'],
                shuffle=args['SHUFFLE'],
                num_workers=args['NUM_WORKERS']
            )
            
            outputs, results_list, best_threshold = run_tests(
                model, 
                testing_loader, 
                args['FIND_THRESHOLD'] and not args['ONLY_TEST'],
                args['EVALUATION_THRESHOLD'], 
                args['ONLY_TEST'], 
                language, 
                results_list
            )
            path = f'{path_to_folder}/{language}_{best_threshold}.txt'
            create_submission_txt(outputs, lang_df, path, valid_techniques)

    if not args['ONLY_TEST']:
        df_results = pd.DataFrame(results_list, columns=['language', 'threshold', 'accuracy', 'F1_micro', 'F1_macro'])
        df_results = df_results.sort_values(by=['language', 'F1_micro'], ascending=False)
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_results)
        
        df_results.to_csv(f'{path_to_folder}/metrics_{model_name}.csv', header=True, index=False)
        print(f'Saved to {path_to_folder}/metrics_{model_name}.csv')
    
    print('Done!')
    return df_results if not args['ONLY_TEST'] else None