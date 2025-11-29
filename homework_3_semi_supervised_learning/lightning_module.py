# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import (
    MeanSquaredError, MeanAbsoluteError, R2Score,
    Accuracy, F1Score
)
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score,
    BinaryAUROC, BinaryAveragePrecision
)
import pytorch_lightning as pl


class SemiSupervisedLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer_type='adamw',
        learning_rate=1e-3,
        optimizer_kwargs=None,
        task_type='multiclass',
        metrics_to_log=None,
        pseudo_label_threshold=0.95,
        consistency_weight=0.3,
        warmup_epochs=10
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.task_type = task_type
        

        self.pseudo_label_threshold = pseudo_label_threshold
        self.consistency_weight = consistency_weight
        self.warmup_epochs = warmup_epochs
        
        if 'weight_decay' not in self.optimizer_kwargs:
             self.optimizer_kwargs['weight_decay'] = 1e-4

        if metrics_to_log is None:
            if task_type == 'regression':
                self.metrics_to_log = ['mse', 'mae', 'rmse', 'r2']
            elif task_type == 'binary_classification':
                self.metrics_to_log = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
            elif task_type == 'multiclass':
                self.metrics_to_log = ['accuracy', 'f1_macro']
            else:
                self.metrics_to_log = []
        else:
            self.metrics_to_log = metrics_to_log
        
        self.metrics = nn.ModuleDict()
        self._setup_metrics()
        
        self.train_losses = []
        self.val_losses = []
        self.save_hyperparameters()
    
    def _setup_metrics(self):
        if self.task_type == 'regression':
            metric_map = {
                'mse': MeanSquaredError(),
                'mae': MeanAbsoluteError(),
                'rmse': MeanSquaredError(squared=False),
                'r2': R2Score()
            }
        
        elif self.task_type == 'binary_classification':
            metric_map = {
                'accuracy': BinaryAccuracy(),
                'precision': BinaryPrecision(),
                'recall': BinaryRecall(),
                'f1': BinaryF1Score(),
                'roc_auc': BinaryAUROC(),
                'pr_auc': BinaryAveragePrecision()
            }
        
        elif self.task_type == 'multiclass':
            num_classes = self.model.output_dim
            metric_map = {
                'accuracy': Accuracy(task='multiclass', num_classes=num_classes),
                'f1_macro': F1Score(task='multiclass', num_classes=num_classes, average='macro')
            }
        else:
            metric_map = {}
        
        for metric_name in self.metrics_to_log:
            if metric_name in metric_map:
                self.metrics[metric_name] = metric_map[metric_name]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        labeled_x, labeled_y = batch['labeled']
        labeled_output = self.forward(labeled_x)
        supervised_loss = self.loss_fn(labeled_output, labeled_y)
        

        total_loss = supervised_loss
        pseudo_labels_used = 0
        
        if batch['unlabeled'] is not None:
            unlabeled_x = batch['unlabeled']
            
            if self.current_epoch >= self.warmup_epochs:
                with torch.no_grad():
                    unlabeled_output = self.forward(unlabeled_x)
                    unlabeled_probs = torch.softmax(unlabeled_output, dim=1)
                    max_probs, pseudo_labels = torch.max(unlabeled_probs, dim=1)
                    
                    confidence_mask = max_probs > self.pseudo_label_threshold
                    
                    if confidence_mask.sum() > 0:
                        confident_x = unlabeled_x[confidence_mask]
                        confident_pseudo_labels = pseudo_labels[confidence_mask]
                        
                        confident_output = self.forward(confident_x)
                        unsupervised_loss = self.loss_fn(confident_output, confident_pseudo_labels)
                        
                        current_weight = self.consistency_weight * min(1.0, (self.current_epoch - self.warmup_epochs) / 50.0)
                        total_loss = supervised_loss + current_weight * unsupervised_loss
                        pseudo_labels_used = confidence_mask.sum().item()
            
            self.log('train_pseudo_labels_used', pseudo_labels_used, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_supervised_loss', supervised_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())
        
        if self.task_type == 'regression':
            self._update_regression_metrics(y, y_hat)
        elif self.task_type == 'binary_classification':
            self._update_binary_classification_metrics(y, y_hat)
        elif self.task_type == 'multiclass':
            self._update_multiclass_metrics(y, y_hat)
        
        return loss
    
    def _update_regression_metrics(self, y_true, y_pred):
        for metric_name, metric in self.metrics.items():
            metric.update(y_pred, y_true)
    
    def _update_binary_classification_metrics(self, y_true, y_pred_logits):
        if len(y_pred_logits.shape) == 1 or y_pred_logits.shape[1] == 1:
            y_pred_proba = torch.sigmoid(y_pred_logits.squeeze())
        else:
            y_pred_proba = torch.softmax(y_pred_logits, dim=1)[:, 1]
        
        y_true_long = y_true.long()
        
        for metric_name, metric in self.metrics.items():
            metric.update(y_pred_proba, y_true_long)
    
    def _update_multiclass_metrics(self, y_true, y_pred_logits):
        for metric_name, metric in self.metrics.items():
            metric.update(y_pred_logits, y_true)
    
    def on_validation_epoch_end(self):
        if len(self.metrics) > 0:
            metrics_str = []
            for metric_name in self.metrics_to_log:
                if metric_name in self.metrics:
                    try:
                        metric_value = self.metrics[metric_name].compute()
                        metric_val_item = metric_value.item() if metric_value.numel() == 1 else float(metric_value)
                        self.log(f'val_{metric_name}', metric_val_item, on_step=False, on_epoch=True, sync_dist=False)
                        metrics_str.append(f"{metric_name}={metric_val_item:.4f}")
                    except (ValueError, RuntimeError):
                        pass
            if metrics_str and self.trainer is not None:
                print(f"Epoch {self.current_epoch}: " + ", ".join(metrics_str))
            
            for metric in self.metrics.values():
                metric.reset()
    
    def configure_optimizers(self):
        optimizer_kwargs = {'lr': self.learning_rate, **self.optimizer_kwargs}
        
        if self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), **optimizer_kwargs)
        elif self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), **optimizer_kwargs)
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.parameters(), **optimizer_kwargs)
        elif self.optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), **optimizer_kwargs)
        elif self.optimizer_type == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), **optimizer_kwargs)
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.optimizer_type}")
            
        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def on_test_start(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.task_type == 'regression':
            self._update_regression_metrics(y, y_hat)
        elif self.task_type == 'binary_classification':
            self._update_binary_classification_metrics(y, y_hat)
        elif self.task_type == 'multiclass':
            self._update_multiclass_metrics(y, y_hat)
        
        return loss
    
    def on_test_epoch_end(self):
        if len(self.metrics) > 0:
            metrics_str = []
            for metric_name in self.metrics_to_log:
                if metric_name in self.metrics:
                    try:
                        metric_value = self.metrics[metric_name].compute()
                        metric_val_item = metric_value.item() if metric_value.numel() == 1 else float(metric_value)
                        self.log(f'test_{metric_name}', metric_val_item, on_step=False, on_epoch=True, sync_dist=False)
                        metrics_str.append(f"{metric_name}={metric_val_item:.4f}")
                    except (ValueError, RuntimeError):
                        pass
            if metrics_str and self.trainer is not None:
                print(f"Test results: " + ", ".join(metrics_str))