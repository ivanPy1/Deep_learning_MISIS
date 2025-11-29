# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class CSVDataset(Dataset):
    def __init__(self, csv_path, has_target=True):
        try:
            self.data = pd.read_csv(csv_path)
            self.has_target = has_target
            
            if has_target:
                self.X = torch.FloatTensor(self.data.drop('target', axis=1).values.astype(np.float32))
                self.y = torch.LongTensor(self.data['target'].values.astype(np.int64))
            else:
                if 'target' in self.data.columns:
                    self.X = torch.FloatTensor(self.data.drop('target', axis=1).values.astype(np.float32))
                else:
                    self.X = torch.FloatTensor(self.data.values.astype(np.float32))
                self.y = None
            
            del self.data
            
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            raise
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        try:
            if self.has_target:
                return self.X[idx], self.y[idx]
            else:
                return self.X[idx]
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")

            if self.has_target:
                return self.X[0], self.y[0]
            else:
                return self.X[0]


class SemiSupervisedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_labeled_csv='train_labeled.csv',
        train_unlabeled_csv='train_unlabeled.csv',
        test_csv='test.csv',
        batch_size=256,
        num_workers=0,
        unlabeled_batch_ratio=2,
        val_size=0.2,
        random_state=42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_labeled_csv = os.path.join(data_dir, train_labeled_csv)
        self.train_unlabeled_csv = os.path.join(data_dir, train_unlabeled_csv)
        self.test_csv = os.path.join(data_dir, test_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unlabeled_batch_ratio = unlabeled_batch_ratio
        self.val_size = val_size
        self.random_state = random_state
        
        self.train_labeled_dataset = None
        self.train_unlabeled_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.input_dim = None
        self.n_classes = None
    
    def setup(self, stage=None):
        try:
            print("Loading labeled dataset...")
            full_labeled_dataset = CSVDataset(self.train_labeled_csv, has_target=True)
            
            sample_x, _ = full_labeled_dataset[0]
            self.input_dim = sample_x.shape[0]
            self.n_classes = len(np.unique(full_labeled_dataset.y.numpy()))
            
            if self.val_size > 0:
                all_labels = full_labeled_dataset.y.numpy()
                
                train_idx, val_idx = train_test_split(
                    range(len(full_labeled_dataset)),
                    test_size=self.val_size,
                    random_state=self.random_state,
                    stratify=all_labels
                )
                
                train_data = full_labeled_dataset.X[train_idx]
                train_labels = full_labeled_dataset.y[train_idx]
                val_data = full_labeled_dataset.X[val_idx]
                val_labels = full_labeled_dataset.y[val_idx]
                
                self.train_labeled_dataset = _TensorDataset(train_data, train_labels)
                self.val_dataset = _TensorDataset(val_data, val_labels)
            else:
                self.train_labeled_dataset = full_labeled_dataset
                self.val_dataset = full_labeled_dataset
            
            print("Loading unlabeled dataset...")
            if os.path.exists(self.train_unlabeled_csv):
                self.train_unlabeled_dataset = CSVDataset(self.train_unlabeled_csv, has_target=False)
            else:
                print(f"Unlabeled dataset not found at {self.train_unlabeled_csv}")
                self.train_unlabeled_dataset = None
                
            print("Loading test dataset...")
            self.test_dataset = CSVDataset(self.test_csv, has_target=True)
            
            print(f'Input dimension: {self.input_dim}')
            print(f'Number of classes: {self.n_classes}')
            print(f'Labeled train samples: {len(self.train_labeled_dataset)}')
            print(f'Validation samples: {len(self.val_dataset)}')
            if self.train_unlabeled_dataset:
                print(f'Unlabeled train samples: {len(self.train_unlabeled_dataset)}')
            print(f'Test samples: {len(self.test_dataset)}')
            
        except Exception as e:
            print(f"Error in setup: {e}")
            raise
    
    def train_dataloader(self):
        labeled_loader = DataLoader(
            self.train_labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False if self.num_workers == 0 else True,
        )
        
        if self.train_unlabeled_dataset and len(self.train_unlabeled_dataset) > 0:
            unlabeled_loader = DataLoader(
                self.train_unlabeled_dataset,
                batch_size=self.batch_size * self.unlabeled_batch_ratio,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                persistent_workers=False if self.num_workers == 0 else True,
            )
            return {'labeled': labeled_loader, 'unlabeled': unlabeled_loader}
        else:
            return {'labeled': labeled_loader, 'unlabeled': None}
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )


class _TensorDataset(Dataset):
    """Простой датасет из тензоров"""
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.has_target = y is not None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.has_target:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]