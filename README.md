# Gradient Flush at Slavic NLP 2025 Task: Leveraging Slavic BERT for Persuasion Technique Classification

This repository contains supplementary material for the paper *Gradient Flush at Slavic NLP 2025 Task: Leveraging Slavic BERT for Persuasion Technique Classification*.

## Abstract

TODO

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