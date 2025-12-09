import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: ANALYZE MISSING DATA PATTERNS
# ============================================================================

def analyze_missing_data(df):
    """
    Analyze and print where each column starts having valid data
    """
    print("=" * 80)
    print("MISSING DATA ANALYSIS")
    print("=" * 80)
    
    results = {}
    for col in df.columns:
        # Find first non-NaN index
        first_valid = df[col].first_valid_index()
        if first_valid is None:
            leading_nans = len(df)
        else:
            leading_nans = df.index.get_loc(first_valid)
        
        results[col] = leading_nans
        status = "Starts with data (no leading NaNs)" if leading_nans == 0 else str(leading_nans)
        print(f"{col}: {status}")
    
    print("\n" + "=" * 80)
    print(f"Total rows in dataset: {len(df)}")
    print("=" * 80)
    
    return results


# ============================================================================
# PART 2: NOVEL DEEP LEARNING IMPUTATION ARCHITECTURE
# ============================================================================

class TemporalAttention(nn.Module):
    """
    Attention mechanism to weigh temporal information for imputation
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_dim)
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        return context, attn_weights


class MultiTaskImputationNetwork(nn.Module):
    """
    Novel architecture combining:
    1. Bidirectional LSTM for temporal patterns
    2. Attention mechanism for important time steps
    3. Feature-specific decoders for each feature group
    4. Residual connections for stable training
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: Bidirectional LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Shared decoder that outputs the full feature vector
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _make_decoder(self, input_dim, output_dim):
        """Create feature-specific decoder with residual connection"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, output_dim)
        )
    
    def forward(self, x, mask):
        # x shape: (batch, seq_len, features)
        # mask shape: (batch, seq_len, features) - 1 where data exists, 0 where missing
        
        # Replace missing values with 0 for input
        x_masked = x * mask
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x_masked)  # (batch, seq_len, hidden_dim*2)
        
        # Attention-weighted context
        context, attn_weights = self.attention(lstm_out)
        context = self.dropout(context)
        
        # Decode full feature vector directly
        imputed = self.output_head(context)
        return imputed, attn_weights


class TimeSeriesDataset(Dataset):
    """Dataset for time series imputation"""
    def __init__(self, data, window_size=20):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        
        # Create mask: 1 where data exists, 0 where NaN
        mask = (~np.isnan(window)).astype(np.float32)
        
        # Replace NaN with 0 for input
        window_filled = np.nan_to_num(window, 0)
        
        # Target is the last row of the window
        target = window_filled[-1]
        target_mask = mask[-1]
        
        return (
            torch.FloatTensor(window_filled),
            torch.FloatTensor(mask),
            torch.FloatTensor(target),
            torch.FloatTensor(target_mask)
        )


# ============================================================================
# PART 3: TRAINING AND IMPUTATION PIPELINE
# ============================================================================

class DeepImputer:
    """
    Complete imputation pipeline with training and inference
    """
    def __init__(self, feature_cols, window_size=20, hidden_dim=128,
                 winsor_lower=0.01, winsor_upper=0.99):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.winsor_lower = winsor_lower
        self.winsor_upper = winsor_upper
        self.winsor_limits = {}
        self.column_medians = {}

    def _compute_winsor_limits(self, df):
        """Compute per-column winsorization bounds, skipping categorical D-columns."""
        limits = {}
        for col in df.columns:
            if col.startswith('D'):
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            lower = series.quantile(self.winsor_lower)
            upper = series.quantile(self.winsor_upper)
            limits[col] = (lower, upper)
        self.winsor_limits = limits

    def _apply_winsorization(self, df):
        """Apply stored winsor limits to a feature DataFrame copy."""
        df_winsor = df.copy()
        if not self.winsor_limits:
            self._compute_winsor_limits(df_winsor)
        for col, (lower, upper) in self.winsor_limits.items():
            if col in df_winsor.columns:
                df_winsor[col] = df_winsor[col].clip(lower=lower, upper=upper)
        return df_winsor
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Extract feature columns and clip outliers (exclude D-prefixed categorical columns)
        feature_df = df[self.feature_cols].copy()
        feature_df = self._apply_winsorization(feature_df)
        if not self.column_medians:
            medians = feature_df.median(numeric_only=True)
            self.column_medians = medians.to_dict()
        data = feature_df.values
        
        # Fit scaler only on non-NaN values
        non_nan_mask = ~np.isnan(data)
        flat_data = data[non_nan_mask].reshape(-1, 1)
        self.scaler.fit(flat_data)
        
        # Scale data while preserving NaN
        scaled_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            col_data = data[:, i].reshape(-1, 1)
            mask = ~np.isnan(col_data)
            if mask.any():
                transformed = self.scaler.transform(col_data[mask].reshape(-1, 1)).flatten()
                scaled_data[:, i] = np.nan
                scaled_data[mask.flatten(), i] = transformed
            else:
                scaled_data[:, i] = np.nan
        
        return scaled_data
    
    def train(self, df, epochs=50, batch_size=32, lr=0.001):
        """Train the imputation model"""
        print("\n" + "=" * 80)
        print("TRAINING DEEP IMPUTATION MODEL")
        print("=" * 80)
        
        # Prepare data
        scaled_data = self.prepare_data(df)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(scaled_data, self.window_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = len(self.feature_cols)
        self.model = MultiTaskImputationNetwork(input_dim, self.hidden_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(
                dataloader,
                desc=f"Imputer Epoch {epoch + 1}/{epochs}",
                leave=False
            )

            for batch_window, batch_mask, batch_target, batch_target_mask in progress_bar:
                batch_window = batch_window.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_target = batch_target.to(self.device)
                batch_target_mask = batch_target_mask.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                imputed, _ = self.model(batch_window, batch_mask)
                
                # Calculate loss only on originally present values
                loss_per_element = criterion(imputed, batch_target)
                loss = (loss_per_element * batch_target_mask).sum() / batch_target_mask.sum()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / max(num_batches, 1)
            scheduler.step(avg_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.6f}")
        
        print("Training completed!")
        
    def impute(self, df):
        """Impute missing values in the dataframe"""
        print("\n" + "=" * 80)
        print("IMPUTING MISSING VALUES")
        print("=" * 80)
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Prepare data
        scaled_data = self.prepare_data(df)
        imputed_data = scaled_data.copy()
        
        # Impute missing values using bidirectional passes for better context propagation
        # Pass 1: Backward (N → 0) - leverage dense late data to inform sparse early region
        print("  Pass 1/2: Backward imputation...")
        with torch.no_grad():
            for i in tqdm(range(len(scaled_data) - 1, -1, -1), desc="Backward pass", leave=False):
                end_idx = min(len(scaled_data), i + self.window_size)
                window = imputed_data[i:end_idx].copy()
                
                # Pad at end if necessary
                if len(window) < self.window_size:
                    pad_size = self.window_size - len(window)
                    window = np.vstack([window, np.zeros((pad_size, window.shape[1]))])
                
                # Build mask from original scaled_data
                orig_window = scaled_data[i:end_idx]
                if len(orig_window) < self.window_size:
                    pad_size = self.window_size - len(orig_window)
                    orig_window = np.vstack([orig_window, np.zeros((pad_size, orig_window.shape[1]))])
                mask = (~np.isnan(orig_window)).astype(np.float32)
                
                window_filled = np.nan_to_num(window, 0)
                
                window_tensor = torch.FloatTensor(window_filled).unsqueeze(0).to(self.device)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                
                imputed, _ = self.model(window_tensor, mask_tensor)
                imputed_row = imputed.cpu().numpy().flatten()
                
                missing_mask = np.isnan(imputed_data[i])
                imputed_data[i, missing_mask] = imputed_row[missing_mask]
        
        # Pass 2: Forward (0 → N) - refine with forward context
        print("  Pass 2/2: Forward imputation...")
        with torch.no_grad():
            for i in tqdm(range(len(scaled_data)), desc="Forward pass", leave=False):
                start_idx = max(0, i - self.window_size + 1)
                window = imputed_data[start_idx:i + 1].copy()
                
                # Pad at start if necessary
                if len(window) < self.window_size:
                    pad_size = self.window_size - len(window)
                    window = np.vstack([np.zeros((pad_size, window.shape[1])), window])
                
                # Build mask from original scaled_data
                orig_window = scaled_data[start_idx:i + 1]
                if len(orig_window) < self.window_size:
                    pad_size = self.window_size - len(orig_window)
                    orig_window = np.vstack([np.zeros((pad_size, orig_window.shape[1])), orig_window])
                mask = (~np.isnan(orig_window)).astype(np.float32)
                
                window_filled = np.nan_to_num(window, 0)
                
                window_tensor = torch.FloatTensor(window_filled).unsqueeze(0).to(self.device)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                
                imputed, _ = self.model(window_tensor, mask_tensor)
                imputed_row = imputed.cpu().numpy().flatten()
                
                # Average with existing imputation if already filled, otherwise just fill
                still_nan = np.isnan(scaled_data[i])
                imputed_data[i, still_nan] = (imputed_data[i, still_nan] + imputed_row[still_nan]) / 2
        
        # Inverse transform
        final_data = np.zeros_like(imputed_data)
        for i in range(imputed_data.shape[1]):
            col_data = imputed_data[:, i].reshape(-1, 1)
            final_data[:, i] = self.scaler.inverse_transform(col_data).flatten()
        
        # Create imputed dataframe and ensure winsorization consistency
        final_df = pd.DataFrame(final_data, columns=self.feature_cols, index=df.index)
        final_df = self._apply_winsorization(final_df)

        # Fallback: fill any residual NaNs with learned medians
        if final_df.isna().any().any():
            for col in self.feature_cols:
                if final_df[col].isna().any():
                    fallback_value = self.column_medians.get(col)
                    if fallback_value is None or np.isnan(fallback_value):
                        fallback_value = 0.0
                    final_df[col] = final_df[col].fillna(fallback_value)
        df_imputed = df.copy()
        df_imputed[self.feature_cols] = final_df
        
        remaining_nans = df_imputed[self.feature_cols].isna().sum().sum()
        print(f"Imputation completed! Remaining NaN: {remaining_nans}")
        
        return df_imputed


# ============================================================================
# PART 4: USAGE EXAMPLE
# ============================================================================

def main(csv_path):
    """
    Main pipeline to analyze and impute time series data
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Analyze missing data
    missing_analysis = analyze_missing_data(df)
    
    # Identify feature columns (exclude date_id, forward_returns, etc.)
    feature_cols = [col for col in df.columns if col not in 
                   ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]
    
    print(f"\nFeature columns to impute: {len(feature_cols)}")
    print(f"Columns: {feature_cols[:10]}... (showing first 10)")
    
    # Initialize and train imputer
    imputer = DeepImputer(feature_cols, window_size=30, hidden_dim=128)
    imputer.train(df, epochs=5, batch_size=64, lr=0.001)
    
    # Impute missing values
    df_imputed = imputer.impute(df)
    
    # Visualize imputation quality (for a sample column)
    sample_col = 'E1'
    if sample_col in feature_cols:
        plt.figure(figsize=(15, 5))
        plt.plot(df[sample_col], label='Original (with NaN)', alpha=0.7)
        plt.plot(df_imputed[sample_col], label='Imputed', alpha=0.7)
        plt.title(f'Imputation Quality: {sample_col}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('imputation_quality.png', dpi=300)
        print("\nVisualization saved as 'imputation_quality.png'")
    
    # Save imputed data
    df_imputed.to_csv('imputed_data.csv', index=False)
    print("\nImputed data saved as 'imputed_data.csv'")
    
    return df_imputed, imputer


# Example usage:
# df_imputed, imputer = main('your_data.csv')