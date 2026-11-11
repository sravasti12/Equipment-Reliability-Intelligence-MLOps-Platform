"""A reproducible unidirectional LSTM for causal RUL sequence regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CausalLSTM(nn.Module):
    """Single-layer unidirectional LSTM that regresses from each window end."""

    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(sequence)
        output: torch.Tensor = self.output(hidden[-1]).squeeze(-1)
        return output


@dataclass(frozen=True)
class LSTMModel:
    """Fitted sequence model together with its train-fitted preprocessing."""

    model: CausalLSTM
    preprocessor: ColumnTransformer
    window_size: int
    target_scale: float
    device: torch.device


def _dense_features(
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
) -> npt.NDArray[np.float32]:
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)


def causal_sequences(
    values: npt.NDArray[np.float32],
    metadata: pd.DataFrame,
    *,
    window_size: int,
) -> npt.NDArray[np.float32]:
    """Build left-padded windows that contain no observations after each row."""
    if window_size < 1 or len(values) != len(metadata):
        raise ValueError("window_size must be positive and rows must be aligned")
    identities = [
        column
        for column in ("source", "dataset_split", "machine_id")
        if column in metadata
    ]
    if "machine_id" not in identities:
        raise ValueError("metadata must include machine_id")
    result = np.empty((len(values), window_size, values.shape[1]), dtype=np.float32)
    groups = metadata.groupby(identities, sort=False, dropna=False).indices.values()
    for raw_indices in groups:
        indices = np.asarray(raw_indices, dtype=np.intp)
        for position, row_index in enumerate(indices):
            start = max(0, position - window_size + 1)
            preceding = values[indices[start : position + 1]]
            padding = window_size - len(preceding)
            result[row_index, :padding] = preceding[0]
            result[row_index, padding:] = preceding
    return result


def fit_lstm(
    train_features: pd.DataFrame,
    train_target: pd.Series[float],
    train_metadata: pd.DataFrame,
    *,
    window_size: int,
    epochs: int,
    random_seed: int,
    batch_size: int = 512,
    hidden_size: int = 32,
) -> LSTMModel:
    """Fit an LSTM using a deterministic CPU training loop."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    from reliability_platform.modeling.baselines import _preprocessor

    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    preprocessor = _preprocessor(train_features, scale_numeric=True)
    preprocessor.fit(train_features)
    train_values = _dense_features(preprocessor, train_features)
    train_sequences = causal_sequences(
        train_values, train_metadata, window_size=window_size
    )
    target_scale = max(float(train_target.max()), 1.0)
    targets = train_target.to_numpy(dtype=np.float32) / target_scale
    dataset = TensorDataset(
        torch.from_numpy(train_sequences),
        torch.from_numpy(targets),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = CausalLSTM(train_values.shape[1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()
    return LSTMModel(
        model=model,
        preprocessor=preprocessor,
        window_size=window_size,
        target_scale=target_scale,
        device=torch.device("cpu"),
    )


def predict_lstm(
    fitted: LSTMModel,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> npt.NDArray[np.float64]:
    """Predict non-negative cycle-level RUL from current/past sequence windows."""
    values = _dense_features(fitted.preprocessor, features)
    sequences = causal_sequences(values, metadata, window_size=fitted.window_size)
    fitted.model.eval()
    with torch.inference_mode():
        predictions = fitted.model(torch.from_numpy(sequences))
    raw_predictions: npt.NDArray[np.float32] = np.asarray(
        predictions.numpy(), dtype=np.float32
    )
    scaled_predictions = raw_predictions.astype(np.float64) * fitted.target_scale
    return np.maximum(scaled_predictions, np.float64(0.0))


def save_lstm_state(fitted: LSTMModel, path: str) -> None:
    """Serialize weights and reproducibility metadata for MLflow artifacts."""
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "preprocessor": fitted.preprocessor,
            "window_size": fitted.window_size,
            "target_scale": fitted.target_scale,
            "hidden_size": fitted.model.recurrent.hidden_size,
        },
        path,
    )
