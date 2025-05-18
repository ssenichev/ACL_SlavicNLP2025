import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from training_evaluation.model_data_preparation import create_dataset_training
import os

def run_training(train_args, eval_args):
    training_set, testing_set = create_dataset_training(
        MODEL_NAME=train_args['MODEL_NAME'],
        train_path=train_args['TRAIN_PATH'],
        test_path=train_args['TEST_PATH'],
        TRAINING_LANGUAGES=train_args['TRAINING_LANGUAGES']
    )

    with open('techniques_subtask3.txt', 'r') as f:
        technique_labels = [line.strip() for line in f.readlines()]
    
    training_loader = DataLoader(
        training_set,
        batch_size=train_args['TRAIN_BATCH_SIZE'],
        shuffle=train_args['TRAIN_SHUFFLE'],
        num_workers=train_args['TRAIN_NUM_WORKERS']
    )

    test_loader = DataLoader(
        testing_set,
        batch_size=train_args['TEST_BATCH_SIZE'],
        shuffle=train_args['TEST_SHUFFLE'],
        num_workers=train_args['TEST_NUM_WORKERS']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({"device": device})
    
    lm = LanguageModel(
        training_loader=training_loader,
        test_loader=test_loader,
        device=device,
        train_args=train_args,
        eval_args=eval_args,
        technique_labels=technique_labels
    )
    return lm.train_over_epochs()

class LanguageModel:
    def __init__(self, training_loader, test_loader, device, train_args, eval_args, technique_labels):
        self.train_args = train_args
        self.eval_args = eval_args
        self.device = device
        self.test_loader = test_loader
        self.training_loader = training_loader
        self.technique_labels = technique_labels
        self.threshold = eval_args['EVALUATION_THRESHOLD']
        
        # Initialize model
        model_class = self._get_model_class()
        self.model = model_class(num_labels=len(technique_labels))
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=self.train_args['LEARNING_RATE']
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_args['EPOCHS'],
            eta_min = self.train_args['LEARNING_RATE'] / 100,
            last_epoch=-1
        )
        
        if train_args.get('FREEZE_LAYERS'):
            self._freeze_layers(train_args['FREEZE_LAYERS'])
        
        wandb.watch(self.model, log="all", log_freq=100)

    def _get_model_class(self):
        from training_evaluation.models import (
            BERTBase, EnglishRobertaBase, EnglishRobertaLarge,
            mBERTBase, XLMRobertaBase, XLMRobertaLarge, 
            XLMRobertaL_ParlaMint, mBERT_Slavic
        )
        
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
        
        model_class = model_classes.get(self.train_args['MODEL_NAME'])
        if not model_class:
            raise ValueError(f"Unsupported model: {self.train_args['MODEL_NAME']}")
        return model_class

    def _compute_metrics(self, outputs, targets):
        predictions = (torch.sigmoid(outputs) > self.threshold).float()
        return {
            'f1_micro': f1_score(targets, predictions, average='micro'),
            'f1_macro': f1_score(targets, predictions, average='macro'),
            'accuracy': accuracy_score(targets, predictions)
        }

    def _log_prediction_examples(self, outputs, targets, prefix="train"):
        predictions = (torch.sigmoid(outputs) > self.threshold).float()
        
        report = classification_report(
            targets, predictions,
            target_names=self.technique_labels,
            zero_division=0,
            output_dict=True
        )
        
        for technique in self.technique_labels:
            metrics = report[technique]
            wandb.log({
                f"{prefix}/class/{technique}/precision": metrics['precision'],
                f"{prefix}/class/{technique}/recall": metrics['recall'],
                f"{prefix}/class/{technique}/f1": metrics['f1-score'],
                f"{prefix}/class/{technique}/support": metrics['support']
            })

        examples = []
        for i in range(min(5, len(predictions))):
            pred_techniques = [self.technique_labels[j] for j in range(len(predictions[i])) if predictions[i][j] == 1]
            true_techniques = [self.technique_labels[j] for j in range(len(targets[i])) if targets[i][j] == 1]
            examples.append([i, pred_techniques, true_techniques])
        
        pred_table = wandb.Table(
            columns=["id", "predicted_techniques", "true_techniques"],
            data=examples
        )
        wandb.log({f"{prefix}_prediction_examples": pred_table})

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        for data in tqdm(self.training_loader, desc=f"Epoch {epoch} Training"):
            ids = data['ids'].to(self.device, dtype=torch.long)
            mask = data['mask'].to(self.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            outputs = self.model(ids, mask, token_type_ids)
            loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_outputs.extend(outputs.cpu().detach().numpy())
            all_targets.extend(targets.cpu().numpy())

        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)
        metrics = self._compute_metrics(torch.tensor(all_outputs), all_targets)
        
        metrics.update({
            'train/loss': total_loss / len(self.training_loader),
            'epoch': epoch
        })
        
        wandb.log(metrics)
        self._log_prediction_examples(torch.tensor(all_outputs), all_targets, prefix="train")
        
        return metrics['train/loss']

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc=f"Epoch {epoch} Evaluation"):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)
                loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
                
                total_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)
        metrics = self._compute_metrics(torch.tensor(all_outputs), all_targets)
        
        metrics.update({
            'val/loss': total_loss / len(self.test_loader),
            'epoch': epoch
        })
        
        wandb.log(metrics)
        self._log_prediction_examples(torch.tensor(all_outputs), all_targets, prefix="val")
        
        return metrics['val/loss']

    def save_model(self, epoch, is_best=False):
        print('\nSaving model...')
        prefix = "best_" if is_best else ""
        os.makedirs(f"{self.train_args['MODELS_DIR']}{self.train_args['MODEL_NAME_PREFIX']}/", exist_ok=True)
        model_path = f"{self.train_args['MODELS_DIR']}{self.train_args['MODEL_NAME_PREFIX']}/{prefix}{self.train_args['MODEL_NAME_PREFIX']}_epoch{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.train_args
        }, model_path)
        
        return model_path

    def train_over_epochs(self):
        best_loss = float('inf')
        best_model_path = None
        
        for epoch in range(self.train_args['EPOCHS']):
            train_loss = self.train(epoch)
            val_loss = self.evaluate(epoch)
            
            self.scheduler.step()
            
            regular_model_path = self.save_model(epoch, is_best=False)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = self.save_model(epoch, is_best=True)
                wandb.run.summary["best_val_loss"] = best_loss
                print(f"New best model saved with loss: {best_loss:.4f}")
        
        return best_model_path