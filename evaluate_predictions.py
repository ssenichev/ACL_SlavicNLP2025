import os
import argparse
from submission_evaluator import evaluate_subtask_1, evaluate_subtask_2
from conformity_checker import check_subtask_1_format, check_subtask_2_format

import warnings
warnings.filterwarnings('always')
warnings.simplefilter('always')

def evaluate_predictions(gold_file, pred_file, subtask, per_class_results=False):
    """
    Evaluate model predictions using the PersuasionNLPTools.
    
    Args:
        gold_file (str): Path to the gold standard file
        pred_file (str): Path to the predictions file
        subtask (str): Either 'subtask1' or 'subtask2'
        per_class_results (bool): Whether to save per-class results (only for subtask2)
    """
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"Gold standard file not found: {gold_file}")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    # First check format conformity
    print("Checking format conformity...")
    try:
        if subtask == 'subtask1':
            check_subtask_1_format(pred_file)
            evaluate_subtask_1(gold_file, pred_file)
        else:
            check_subtask_2_format(pred_file)
            evaluate_subtask_2(gold_file, pred_file, per_class_results)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions using PersuasionNLPTools")
    parser.add_argument("--gold", required=True, help="Path to the gold standard file")
    parser.add_argument("--pred", required=True, help="Path to the predictions file")
    parser.add_argument("--subtask", choices=["subtask1", "subtask2"], required=True, 
                       help="Which subtask to evaluate (subtask1 for binary, subtask2 for multi-label)")
    parser.add_argument("--per-class", action="store_true", 
                       help="Save per-class results (only for subtask2)")
    
    args = parser.parse_args()
    
    evaluate_predictions(args.gold, args.pred, args.subtask, args.per_class) 