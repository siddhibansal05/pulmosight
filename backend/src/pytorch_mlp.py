"""
PyTorch MLP Classifier with Early Stopping, Validation Split, and Torch-Free NumPy Serving
(`backend/src/pytorch_mlp.py`).

Trains using PyTorch (`torch.nn.Module`, `BCEWithLogitsLoss`, AdamW, internal 15% stratified
validation split with early stopping). After training, extracts the learned layer weights
and BatchNorm running statistics into pure NumPy arrays (`self.numpy_layers_`) and drops the
PyTorch `nn.Module` reference before serialization so `joblib.load()` and `predict_proba()`
require ZERO `torch` dependency at serving time.
"""

import copy
from typing import Dict, List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from backend.src.config import RANDOM_SEED


class PyTorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible PyTorch Multi-Layer Perceptron binary classifier
    with internal validation split, early stopping, and torch-free NumPy inference.
    Note: `ClassifierMixin` precedes `BaseEstimator` in MRO for scikit-learn >= 1.6 tags.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.20,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 80,
        patience: int = 12,
        val_fraction: float = 0.15,
        random_state: int = RANDOM_SEED,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def _build_network(self, input_dim: int):
        import torch.nn as nn

        layers = []
        in_d = input_dim
        for h_d in self.hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _export_numpy_layers(net) -> List[Dict[str, np.ndarray]]:
        """Extract eval-mode Linear and BatchNorm1d parameters into pure NumPy arrays."""
        import torch.nn as nn

        exported = []
        for module in net:
            if isinstance(module, nn.Linear):
                exported.append(
                    {
                        "type": "linear",
                        "weight": module.weight.detach().cpu().numpy().astype(np.float32),
                        "bias": module.bias.detach().cpu().numpy().astype(np.float32),
                    }
                )
            elif isinstance(module, nn.BatchNorm1d):
                exported.append(
                    {
                        "type": "batchnorm1d",
                        "gamma": module.weight.detach().cpu().numpy().astype(np.float32),
                        "beta": module.bias.detach().cpu().numpy().astype(np.float32),
                        "running_mean": module.running_mean.detach().cpu().numpy().astype(np.float32),
                        "running_var": module.running_var.detach().cpu().numpy().astype(np.float32),
                        "eps": float(module.eps),
                    }
                )
            elif isinstance(module, nn.ReLU):
                exported.append({"type": "relu"})
        return exported

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X_arr.shape[1]

        if len(np.unique(y_arr)) > 1 and len(y_arr) >= 40:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_arr,
                y_arr,
                test_size=self.val_fraction,
                stratify=y_arr,
                random_state=self.random_state,
            )
        else:
            X_tr, X_val, y_tr, y_val = X_arr, X_arr, y_arr, y_arr

        n_pos = float(np.sum(y_tr == 1))
        n_neg = float(np.sum(y_tr == 0))
        pos_weight_val = n_neg / max(1.0, n_pos)

        net = self._build_network(self.n_features_in_)
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1)
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )

        X_val_t = torch.from_numpy(X_val)
        best_val_auc = -1.0
        best_state = copy.deepcopy(net.state_dict())
        epochs_no_improve = 0
        self.best_epoch_ = 0

        for epoch in range(1, self.max_epochs + 1):
            net.train()
            for bx, by in loader:
                if len(bx) == 1:
                    continue
                optimizer.zero_grad()
                logits = net(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_logits = net(X_val_t).squeeze(1)
                val_probs = torch.sigmoid(val_logits).numpy()

            val_auc = (
                float(roc_auc_score(y_val, val_probs))
                if len(np.unique(y_val)) > 1
                else 0.5
            )

            if val_auc > best_val_auc + 1e-4:
                best_val_auc = val_auc
                best_state = copy.deepcopy(net.state_dict())
                self.best_epoch_ = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        # Export to pure NumPy arrays so serving/deserialization requires zero torch dependency
        self.numpy_layers_ = self._export_numpy_layers(net)
        self.best_val_auc_ = round(float(best_val_auc), 4)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Pure NumPy forward pass (zero torch dependency at serving time)."""
        h = np.asarray(X, dtype=np.float32)
        for layer in self.numpy_layers_:
            ltype = layer["type"]
            if ltype == "linear":
                h = h @ layer["weight"].T + layer["bias"]
            elif ltype == "batchnorm1d":
                h = (h - layer["running_mean"]) / np.sqrt(
                    layer["running_var"] + layer["eps"]
                )
                h = h * layer["gamma"] + layer["beta"]
            elif ltype == "relu":
                h = np.maximum(0.0, h)
        logits = np.clip(h.reshape(-1), -30.0, 30.0)
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        probs_pos = np.clip(probs_pos, 1e-5, 1.0 - 1e-5)
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
