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
        optimizer_type='adam',
        learning_rate=1e-3,
        optimizer_kwargs=None,
        task_type='regression',
        metrics_to_log=None,
        use_pseudo_labeling=True,
        consistency_weight=0.3,
        temperature=0.5
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.task_type = task_type
        self.use_pseudo_labeling = use_pseudo_labeling
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        
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
        self.consistency_losses = []
    
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
        if isinstance(batch, list) and len(batch) == 2:
            # Semi-supervised режим
            labeled_batch, unlabeled_batch = batch
            x_labeled, y_labeled = labeled_batch
            x_unlabeled = unlabeled_batch
            
            # Обучение на размеченных данных
            y_hat_labeled = self.forward(x_labeled)
            supervised_loss = self.loss_fn(y_hat_labeled, y_labeled)
            
            # Псевдо-разметка для неразмеченных данных
            consistency_loss = 0
            if self.use_pseudo_labeling and x_unlabeled.shape[0] > 0:
                with torch.no_grad():
                    logits_unlabeled = self.forward(x_unlabeled)
                    pseudo_labels = torch.softmax(logits_unlabeled / self.temperature, dim=1)
                    confident_mask = (pseudo_labels.max(dim=1)[0] > 0.7)
                    
                    if confident_mask.sum() > 0:
                        x_confident = x_unlabeled[confident_mask]
                        pseudo_labels_confident = pseudo_labels[confident_mask]
                        
                        # Аугментация - добавление шума
                        x_augmented = x_confident + torch.randn_like(x_confident) * 0.1
                        logits_augmented = self.forward(x_augmented)
                        
                        consistency_loss = nn.functional.kl_div(
                            nn.functional.log_softmax(logits_augmented / self.temperature, dim=1),
                            pseudo_labels_confident,
                            reduction='batchmean'
                        )
            
            total_loss = supervised_loss + self.consistency_weight * consistency_loss
            
            self.log('train_supervised_loss', supervised_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_consistency_loss', consistency_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            
        else:
            # Обычный supervised режим
            x, y = batch
            y_hat = self.forward(x)
            total_loss = self.loss_fn(y_hat, y)
            self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
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
    
    def configure_optimizers(self):
        optimizer_kwargs = {
            'lr': self.learning_rate,
            'weight_decay': 1e-4,  # Добавим L2 регуляризацию
            **self.optimizer_kwargs
        }
        
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
        
        # Добавим scheduler для лучшей сходимости
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
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