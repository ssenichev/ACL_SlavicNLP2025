# Gradient Flush at Slavic NLP 2025 Task: Leveraging Slavic BERT and Translation for Persuasion Techniques Classification

## Abstract

The task of persuasion techniques detection is limited by several challenges, such as insufficient training data and ambiguity in labels. In this paper, we describe a solution for the Slavic NLP 2025 Shared Task. It utilizes multilingual XLM-RoBERTa, that was trained on 100 various languages, and Slavic BERT, a model fine-tuned on four languages of the Slavic group. We suggest to augment the training dataset with related data from previous shared tasks, as well as some automatic translations from English and German. The resulting solutions are ranked among the top 3 for Russian in the Subtask 1 and for all languages in the Subtask 2. We release the code for our solution.

## Structure of code

This repository is structured in three folders:

1. [Input data](_input_data) – raw data provided by organizers
2. [Processed data](_processed_data) - processed datasets
3. [Training and evaluation scripts](training_evaluation) - scripts for training and evaluation


## Running the code
1. Run the main script `run_persuasion_detection.py` to train and evaluate the model
2. Configure the training parameters in `args_train` dictionary:
   - Set model architecture and hyperparameters
   - Specify input/output paths
   - Adjust batch sizes and learning rate
3. Configure evaluation parameters in `args_eval` dictionary:
   - Set evaluation thresholds
   - Specify languages to evaluate
   - Define output directories
4. The script will:
   - Train the model and save it to the specified location
   - Generate evaluation results

## Citing the Paper:
```
TODO
```
