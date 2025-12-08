"""Minimal sequential training script for Hull Tactical Market Prediction."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from metric import ParticipantVisibleError, score


LABEL_COLUMNS = [
	"forward_returns",
	"risk_free_rate",
	"market_forward_excess_returns",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--train-path", default="train.csv", type=Path)
	parser.add_argument("--output-path", default="submission.csv", type=Path)
	parser.add_argument("--sequence-length", default=16, type=int)
	parser.add_argument("--hidden-dim", default=256, type=int)
	parser.add_argument("--num-layers", default=8, type=int)
	parser.add_argument("--dropout", default=0.4, type=float)
	parser.add_argument("--epochs", default=10, type=int)
	parser.add_argument("--batch-size", default=128, type=int)
	parser.add_argument("--lr", default=2e-3, type=float)
	parser.add_argument("--clip-lower", default=0.01, type=float)
	parser.add_argument("--clip-upper", default=0.99, type=float)
	parser.add_argument("--temperature", default=0.01, type=float)
	parser.add_argument("--val-ratio", default=0.15, type=float)
	parser.add_argument("--test-ratio", default=0.1, type=float)
	parser.add_argument("--patience", default=5, type=int)
	parser.add_argument("--seed", default=7, type=int)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_frame(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if "date_id" not in df.columns:
		raise ValueError("Expected a date_id column for chronological sorting")
	df = df.sort_values("date_id").reset_index(drop=True)
	df = df.ffill().bfill().fillna(0)
	return df


def compute_winsor_bounds(df: pd.DataFrame, lower: float, upper: float) -> Tuple[pd.Series, pd.Series]:
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	lower_bounds = df[numeric_cols].quantile(lower)
	upper_bounds = df[numeric_cols].quantile(upper)
	return lower_bounds, upper_bounds


def apply_winsor(df: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
	cols = [c for c in df.columns if c in lower.index]
	clipped = df.copy()
	clipped[cols] = clipped[cols].clip(lower=lower[cols], upper=upper[cols], axis=1)
	return clipped


def standardize(train_df: pd.DataFrame, other_df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	means = train_df[feature_cols].mean()
	stds = train_df[feature_cols].std().replace(0, 1.0)
	train_df.loc[:, feature_cols] = (train_df[feature_cols] - means) / stds
	other_df.loc[:, feature_cols] = (other_df[feature_cols] - means) / stds
	return train_df, other_df


class SequenceDataset(Dataset):
	def __init__(
		self,
		feature_array: np.ndarray,
		targets: np.ndarray,
		seq_len: int,
		start: int,
		end: int,
		return_indices: bool = False,
	) -> None:
		self.features = feature_array
		self.targets = targets
		self.seq_len = seq_len
		self.start = start
		self.end = end
		self.return_indices = return_indices

	def __len__(self) -> int:
		return self.end - self.start

	def __getitem__(self, idx: int):  # type: ignore[override]
		base_idx = self.start + idx
		x = self.features[base_idx : base_idx + self.seq_len]
		target_pos = base_idx + self.seq_len
		y = self.targets[target_pos]
		sample = torch.from_numpy(x)
		target = torch.tensor(y, dtype=torch.float32)
		if self.return_indices:
			return sample, target, target_pos
		return sample, target


class SequenceWindowDataset(Dataset):
	def __init__(self, feature_array: np.ndarray, seq_len: int) -> None:
		self.features = feature_array
		self.seq_len = seq_len

	def __len__(self) -> int:
		return len(self.features) - self.seq_len

	def __getitem__(self, idx: int):  # type: ignore[override]
		x = self.features[idx : idx + self.seq_len]
		target_pos = idx + self.seq_len
		return torch.from_numpy(x), target_pos


class LSTMForecaster(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		proj_dim = max(hidden_dim // 2, 8)
		self.head = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, proj_dim),
			nn.GELU(),
			nn.Linear(proj_dim, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
		seq_out, _ = self.lstm(x)
		last_hidden = seq_out[:, -1, :]
		return self.head(last_hidden).squeeze(-1)


def returns_to_positions(pred_returns: np.ndarray, temperature: float, bias: float = 0.0) -> np.ndarray:
	temperature = max(temperature, 1e-4)
	adjusted = pred_returns + bias
	scaled = 1.0 + np.tanh(adjusted / temperature)
	return np.clip(scaled, 0.0, 2.0)


@dataclass
class TrainingArtifacts:
	model: nn.Module
	feature_array: np.ndarray
	train_len: int
	temperature: float
	position_bias: float = 0.0


def train_model(
	args: argparse.Namespace,
	feature_array: np.ndarray,
	target_array: np.ndarray,
	device: torch.device,
) -> Tuple[TrainingArtifacts, Dict[str, float], DataLoader]:
	total_sequences = len(target_array) - args.sequence_length
	if total_sequences <= 0:
		raise ValueError("Not enough observations for the chosen sequence length")

	val_sequences = max(1, int(total_sequences * args.val_ratio))
	train_sequences = total_sequences - val_sequences
	if train_sequences <= 0:
		raise ValueError("Validation split is too large for the dataset")
	print(
		f"Sequence windows -> train: {train_sequences}, val: {val_sequences}, seq_len: {args.sequence_length}"
	)

	model = LSTMForecaster(
		input_dim=feature_array.shape[1],
		hidden_dim=args.hidden_dim,
		num_layers=args.num_layers,
		dropout=args.dropout,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.MSELoss()

	train_dataset = SequenceDataset(
		feature_array,
		target_array,
		seq_len=args.sequence_length,
		start=0,
		end=train_sequences,
	)
	val_dataset = SequenceDataset(
		feature_array,
		target_array,
		seq_len=args.sequence_length,
		start=train_sequences,
		end=train_sequences + val_sequences,
		return_indices=True,
	)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

	best_val_loss = math.inf
	best_state = None
	patience = max(1, args.patience)
	patience_counter = 0

	for epoch in range(args.epochs):
		model.train()
		train_loss = 0.0
		train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1} train", leave=False)
		for xb, yb in train_iter:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			preds = model(xb)
			loss = criterion(preds, yb)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
			optimizer.step()
			train_loss += loss.item() * len(xb)

		train_loss /= len(train_loader.dataset)

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			val_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1} val", leave=False)
			for xb, yb, _ in val_iter:
				xb = xb.to(device)
				yb = yb.to(device)
				preds = model(xb)
				val_loss += criterion(preds, yb).item() * len(xb)
		val_loss /= len(val_loader.dataset)

		print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = model.state_dict()
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print("Early stopping triggered.")
				break

	if best_state is not None:
		model.load_state_dict(best_state)

	metrics = {"val_loss": best_val_loss}
	artifacts = TrainingArtifacts(
		model=model,
		feature_array=feature_array,
		train_len=len(target_array),
		temperature=args.temperature,
		position_bias=0.0,
	)
	return artifacts, metrics, val_loader


def collect_validation_predictions(
	train_df: pd.DataFrame,
	val_loader: DataLoader,
	model: nn.Module,
	device: torch.device,
) -> Optional[pd.DataFrame]:
	model.eval()
	preds: list[np.ndarray] = []
	indices: list[np.ndarray] = []
	with torch.no_grad():
		for xb, _, idx in tqdm(val_loader, desc="Collecting validation preds", leave=False):
			xb = xb.to(device)
			batch_preds = model(xb).cpu().numpy()
			preds.append(batch_preds)
			indices.append(idx.numpy())

	if not preds:
		return None

	pred_returns = np.concatenate(preds)
	target_indices = np.concatenate(indices)
	cache = train_df.loc[target_indices, ["date_id", "forward_returns", "risk_free_rate"]].reset_index(drop=True)
	cache["raw_pred"] = pred_returns
	return cache


def evaluate_validation_metric(
	val_cache: Optional[pd.DataFrame],
	temperature: float,
	bias: float = 0.0,
) -> Optional[float]:
	if val_cache is None or val_cache.empty:
		return None
	positions = returns_to_positions(val_cache["raw_pred"].to_numpy(), temperature, bias)
	solution = val_cache[["date_id", "forward_returns", "risk_free_rate"]].copy()
	submission = pd.DataFrame({"prediction": positions})
	try:
		metric_value = score(solution, submission, row_id_column_name="date_id")
		return float(metric_value)
	except ParticipantVisibleError as err:
		print(f"Validation metric failed: {err}")
		return None


def search_position_parameters(
	val_cache: Optional[pd.DataFrame],
	base_temperature: float,
) -> Tuple[float, float, Optional[float]]:
	if val_cache is None or val_cache.empty:
		return base_temperature, 0.0, None
	temp_grid = sorted({max(base_temperature * r, 1e-4) for r in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]})
	temp_grid.append(max(1e-4, base_temperature))
	temp_grid = sorted(set(temp_grid))
	bias_grid = np.linspace(-0.2, 0.2, 9)
	best_metric = -math.inf
	best_temp = base_temperature
	best_bias = 0.0
	for temp in temp_grid:
		for bias in bias_grid:
			metric_value = evaluate_validation_metric(val_cache, temp, bias)
			if metric_value is None:
				continue
			if metric_value > best_metric:
				best_metric = metric_value
				best_temp = temp
				best_bias = bias
	if math.isinf(best_metric):
		return base_temperature, 0.0, None
	return best_temp, best_bias, best_metric


def generate_predictions(
	artifacts: TrainingArtifacts,
	feature_array_test: np.ndarray,
	args: argparse.Namespace,
	device: torch.device,
) -> np.ndarray:
	model = artifacts.model
	model.eval()

	combined_features = np.vstack([artifacts.feature_array, feature_array_test])
	window_dataset = SequenceWindowDataset(combined_features, args.sequence_length)
	window_loader = DataLoader(window_dataset, batch_size=args.batch_size, shuffle=False)

	predictions: Dict[int, float] = {}
	with torch.no_grad():
		for xb, positions in tqdm(window_loader, desc="Generating predictions", leave=False):
			xb = xb.to(device)
			preds = model(xb).cpu().numpy()
			for pos, pred in zip(positions.numpy(), preds, strict=False):
				predictions[int(pos)] = float(pred)

	test_start = artifacts.train_len
	test_preds = [predictions.get(idx) for idx in range(test_start, test_start + len(feature_array_test))]
	if any(pred is None for pred in test_preds):
		raise RuntimeError("Missing predictions for part of the test horizon")
	return np.array(test_preds, dtype=np.float32)


def evaluate_holdout_score(holdout_df: pd.DataFrame, positions: np.ndarray) -> Optional[float]:
	solution = holdout_df[["date_id", "forward_returns", "risk_free_rate"]].reset_index(drop=True)
	submission = pd.DataFrame({"prediction": positions})
	try:
		return float(score(solution, submission, row_id_column_name="date_id"))
	except ParticipantVisibleError as err:
		print(f"Holdout metric failed: {err}")
		return None


def plot_allocations(holdout_df: pd.DataFrame, positions: np.ndarray, plot_path: Path) -> None:
	plt.figure(figsize=(10, 4))
	plt.plot(holdout_df["date_id"], positions, label="Model allocation", linewidth=1.5)
	plt.axhline(1.0, color="gray", linestyle=":", label="Baseline allocation = 1")
	plt.xlabel("date_id")
	plt.ylabel("Allocation")
	plt.title("Holdout Allocations vs Baseline")
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=200)
	plt.close()
	print(f"Allocation plot saved to {plot_path}.")


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	full_df = load_frame(args.train_path)
	feature_cols = sorted([c for c in full_df.columns if c not in LABEL_COLUMNS])
	test_ratio = min(max(args.test_ratio, 0.05), 0.5)
	holdout_size = max(int(len(full_df) * test_ratio), args.sequence_length + 1)
	holdout_size = min(holdout_size, len(full_df) - (args.sequence_length + 1))
	train_cutoff = len(full_df) - holdout_size
	if train_cutoff <= args.sequence_length:
		raise ValueError("Not enough history left after reserving the holdout split; reduce --sequence-length or --test-ratio")
	train_df = full_df.iloc[:train_cutoff].reset_index(drop=True)
	holdout_df = full_df.iloc[train_cutoff:].reset_index(drop=True)
	print(
		f"Train rows: {len(train_df)} | Holdout rows: {len(holdout_df)} | Features: {len(feature_cols)}"
	)

	feature_bounds = compute_winsor_bounds(train_df[feature_cols], args.clip_lower, args.clip_upper)
	print(
		"Feature winsor bounds ready (sample):",
		{
			"lower": float(feature_bounds[0].median()),
			"upper": float(feature_bounds[1].median()),
		},
	)
	train_features = apply_winsor(train_df[feature_cols], *feature_bounds)
	holdout_features = apply_winsor(holdout_df[feature_cols], *feature_bounds)

	train_features, holdout_features = standardize(train_features, holdout_features, feature_cols)

	target_bounds = train_df["forward_returns"].quantile([args.clip_lower, args.clip_upper])
	target_array = train_df["forward_returns"].clip(target_bounds.iloc[0], target_bounds.iloc[1]).values.astype(np.float32)
	print(
		f"Forward return clip range: [{target_bounds.iloc[0]:.6f}, {target_bounds.iloc[1]:.6f}]"
	)

	feature_array = train_features.values.astype(np.float32)
	feature_array_test = holdout_features.values.astype(np.float32)
	print(
		f"Feature tensors -> train: {feature_array.shape}, holdout: {feature_array_test.shape}"
	)

	artifacts, metrics, val_loader = train_model(args, feature_array, target_array, device)
	print(f"Best validation loss: {metrics['val_loss']:.6f}")

	val_cache = collect_validation_predictions(train_df, val_loader, artifacts.model, device)
	initial_metric = evaluate_validation_metric(val_cache, artifacts.temperature)
	if initial_metric is not None:
		print(f"Local adjusted Sharpe (validation tail): {initial_metric:.6f}")

	best_temp, best_bias, tuned_metric = search_position_parameters(val_cache, artifacts.temperature)
	if tuned_metric is not None:
		artifacts.temperature = best_temp
		artifacts.position_bias = best_bias
		if initial_metric is None or tuned_metric - initial_metric > 1e-6:
			print(
				f"Tuned position mapping -> temp={best_temp:.6f}, bias={best_bias:+.4f}, metric={tuned_metric:.6f}"
			)
		else:
			print(
				f"Tuned position mapping -> temp={best_temp:.6f}, bias={best_bias:+.4f} (metric unchanged)."
			)

	test_pred_returns = generate_predictions(artifacts, feature_array_test, args, device)
	test_positions = returns_to_positions(
		test_pred_returns,
		artifacts.temperature,
		artifacts.position_bias,
	)

	submission = pd.DataFrame({
		"date_id": holdout_df.loc[:, "date_id"],
		"prediction": test_positions,
	})
	submission.to_csv(args.output_path, index=False)
	print(f"Submission saved to {args.output_path}.")

	proxy = evaluate_holdout_score(holdout_df, test_positions)
	if proxy is not None:
		print(f"Holdout adjusted Sharpe: {proxy:.6f}")

	baseline_positions = np.ones(len(holdout_df), dtype=np.float32)
	baseline_proxy = evaluate_holdout_score(holdout_df, baseline_positions)
	if baseline_proxy is not None:
		print(f"Baseline (allocation=1) Sharpe: {baseline_proxy:.6f}")
		if proxy is not None:
			delta = proxy - baseline_proxy
			print(f"Model vs baseline delta: {delta:+.6f}")

	plot_path = args.output_path.with_suffix(".png")
	plot_allocations(holdout_df, test_positions, plot_path)


if __name__ == "__main__":
	main()
