from training_evaluation.training import run_training
from training_evaluation.evaluating import run_eval
from training_evaluation.models import BERTBase, EnglishRobertaBase, EnglishRobertaLarge, mBERTBase, XLMRobertaBase, XLMRobertaLarge
from evaluate_predictions import evaluate_predictions
import wandb
import os
import glob

from submission_evaluator import evaluate_subtask_1, evaluate_subtask_2
from conformity_checker import check_subtask_1_format, check_subtask_2_format

args_train = {
    'PROJECT_NAME': PROJECT_NAME,
    'RUN_NAME': RUN_NAME,

    'MODEL_NAME_PREFIX': MODEL_NAME_PREFIX,
    'EPOCHS': 10,
    'LEARNING_RATE': 3e-05,
    'MODEL_NAME': 'DeepPavlov/bert-base-bg-cs-pl-ru-cased',
    'DROPOUT_RATE': 0.3,
    'MODELS_DIR': 'models/',
    'TRAIN_PATH': '_processed_data/train_set.csv',
    'TEST_PATH': '_processed_data/test_set.csv',

    'FREEZE_LAYERS': None,

    'TRAIN_BATCH_SIZE': 64,
    'TRAIN_SHUFFLE': True,
    'TRAIN_NUM_WORKERS': 0,
    'CURRENT_EPOCH': 0,
    'CURRENT_TRAIN_LOSS': 0,
    'TRAINING_LANGUAGES': None,
    
    'TEST_BATCH_SIZE': 64,
    'TEST_SHUFFLE': False,
    'TEST_NUM_WORKERS': 0,
    'BEST_LOSS': float('inf'),
    'CURRENT_TEST_LOSS': 0
}

args_eval = {
        'MODEL_FILE_NAME': '',
        'RESULT_DIR_NAME': 'test_results',
        'FIND_THRESHOLD': True,
        'MODEL_NAME': args_train['MODEL_NAME'],
        'ONLY_TEST': True,
        'EVALUATION_SET': '_processed_data/test_set.csv',
        'LANGUAGES': ["SI"],
        'EVALUATION_THRESHOLD': 0.35,
        'BINARY': False,

        'BATCH_SIZE': 64,
        'SHUFFLE': False,
        'NUM_WORKERS': 0
    }

wandb.init(
    project=args_train['PROJECT_NAME'],
    name=args_train['RUN_NAME'],
    config=args_train
)

try:
    model_file = run_training(args_train, args_eval)
    args_eval['MODEL_FILE_NAME'] = model_file
    run_eval(args_eval)

    model_name = args_eval['MODEL_FILE_NAME'].split('/')[-1].split('.')[0]
    path_to_folder = f'{args_eval["RESULT_DIR_NAME"]}/'
    pattern = os.path.join(path_to_folder, model_name)
    path_to_folder = glob.glob(pattern)[0]
    
    prediction_files = [f for f in os.listdir(path_to_folder) if (f.endswith('.txt') and f.startswith(args_eval['LANGUAGES'][0]))]
    
    for pred_file in prediction_files:
        language = pred_file.split('_')[0]
        print(f"\nEvaluating predictions format for language: {language}")
        
        gold_file = args_eval['EVALUATION_SET']
        pred_file = os.path.join(path_to_folder, pred_file)

        print(f"Submission file path: {pred_file}")
        check_subtask_2_format(pred_file)
        evaluate_subtask_2(gold_file, pred_file, language)

finally:
    wandb.finish()