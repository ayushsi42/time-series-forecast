import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ============================================================================
# ADVANCED MARKET PREDICTION ARCHITECTURE
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for capturing feature interactions"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(x + output)


class TemporalConvBlock(nn.Module):
    """Temporal convolutional block for pattern extraction"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.transpose(1, 2)  # (batch, seq_len, features)


class MarketPredictionNetwork(nn.Module):
    """
    Deep architecture for market prediction combining:
    1. Multi-scale temporal convolutions for local patterns
    2. Stacked Transformer-style attention for long-range dependencies
    3. Deep Bidirectional LSTM for sequential modeling
    4. Multiple prediction heads for returns and uncertainty
    """
    def __init__(self, input_dim, hidden_dim=512, num_layers=4, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale temporal convolution blocks (more layers, varying dilations)
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(5)  # 5 layers: dilation 1, 2, 4, 8, 16
        ])
        
        # Stacked multi-head self-attention (2 layers)
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads=8, dropout=dropout)
            for _ in range(2)
        ])
        
        # Deep Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Second LSTM layer for deeper temporal modeling
        self.lstm2 = nn.LSTM(
            hidden_dim, hidden_dim // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Deeper prediction head with residual connections
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Deeper uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Input projection
        x = self.input_proj(x)
        
        # Temporal convolutions with residual connections
        for tcn in self.tcn_blocks:
            x = tcn(x) + x  # Residual connection
        
        # Stacked self-attention
        for attn in self.attention_layers:
            x = attn(x)
        
        # First LSTM
        lstm_out, _ = self.lstm(x)
        
        # Second LSTM for deeper temporal modeling
        lstm_out2, _ = self.lstm2(lstm_out)
        
        # Combine outputs with residual
        combined = lstm_out[:, -1, :] + lstm_out2[:, -1, :]
        
        # Predictions
        returns = self.return_head(combined).squeeze(-1)
        uncertainty = self.uncertainty_head(combined).squeeze(-1)
        
        return returns, uncertainty


class MarketDataset(Dataset):
    """Dataset for market prediction"""
    def __init__(self, features, targets, sequence_length=60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ============================================================================
# BETTING STRATEGY WITH KELLY CRITERION
# ============================================================================

class AdaptiveBettingStrategy:
    """
    Sophisticated betting strategy that:
    1. Uses predicted returns and uncertainty
    2. Applies Kelly Criterion with fractional sizing
    3. Implements volatility targeting
    4. Includes risk management overlays
    """
    def __init__(self, target_vol=0.12, kelly_fraction=0.25, max_leverage=2.0):
        self.target_vol = target_vol  # 120% of market vol = 1.2 * ~0.10
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.market_vol_window = 60  # Days for volatility calculation
        
    def calculate_position(self, predicted_return, uncertainty, market_vol, risk_free_rate):
        """
        Calculate optimal position size
        
        Args:
            predicted_return: Model's return prediction
            uncertainty: Model's uncertainty estimate
            market_vol: Historical market volatility
            risk_free_rate: Current risk-free rate
        
        Returns:
            position: 0-2 where 0=risk-free, 1=100% S&P, 2=200% S&P (leverage)
        """
        # Adjust prediction for uncertainty (more uncertain = more conservative)
        confidence_factor = 1.0 / (1.0 + uncertainty)
        adjusted_return = predicted_return * confidence_factor
        
        # Kelly Criterion: f = (p*b - q) / b
        # Simplified for continuous returns: f = expected_return / variance
        if market_vol > 0:
            kelly_position = (adjusted_return - risk_free_rate) / (market_vol ** 2)
            kelly_position *= self.kelly_fraction  # Fractional Kelly
        else:
            kelly_position = 0
        
        # Volatility targeting: scale position to hit target volatility
        if market_vol > 0:
            vol_scalar = self.target_vol / market_vol
        else:
            vol_scalar = 1.0
        
        # Combine Kelly and vol targeting
        position = kelly_position * vol_scalar
        
        # Risk management overlays
        # 1. No leverage for negative expected returns
        if adjusted_return < risk_free_rate:
            position = min(position, 1.0)
        
        # 2. Reduce position in high uncertainty
        if uncertainty > 0.02:  # High uncertainty threshold
            position *= 0.5
        
        # 3. Clip to valid range [0, max_leverage]
        position = np.clip(position, 0.0, self.max_leverage)
        
        return position


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class MarketPredictor:
    """Complete training and prediction pipeline"""
    def __init__(self, sequence_length=60, hidden_dim=512):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.scaler = RobustScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betting_strategy = AdaptiveBettingStrategy()
        self.feature_cols = None  # Will be set during training
        
    def prepare_features(self, df):
        """Prepare features for training/inference"""
        # Columns to exclude from features
        exclude_cols = [
            'date_id', 'forward_returns', 'risk_free_rate', 
            'market_forward_excess_returns',
            'lagged_forward_returns', 'lagged_risk_free_rate',
            'lagged_market_forward_excess_returns', 'is_scored'
        ]
        
        if self.feature_cols is None:
            # Training mode: learn feature columns
            self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Use only the stored feature columns (handles train/test column mismatch)
        available_cols = [col for col in self.feature_cols if col in df.columns]
        features_df = df[available_cols].copy()
        
        # Calculate rolling statistics
        for window in [5, 10, 20]:
            features_df[f'mean_{window}'] = features_df.rolling(window).mean().mean(axis=1)
            features_df[f'std_{window}'] = features_df.rolling(window).std().mean(axis=1)
        
        # Fill any remaining NaN from rolling calculations
        features_df = features_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return features_df
    
    def train(self, df, epochs=100, batch_size=64, lr=0.0001, test_df=None, eval_every=10):
        """Train the prediction model on full data with periodic test evaluation.
        
        Args:
            df: Training dataframe (imputed)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            test_df: Optional test dataframe for evaluation every eval_every epochs.
                     Must contain 'forward_returns', 'risk_free_rate', 'date_id'.
            eval_every: Evaluate on test set every N epochs (default: 10)
        """
        print("\n" + "=" * 80)
        print("TRAINING MARKET PREDICTION MODEL (DEEP ARCHITECTURE)")
        print("=" * 80)
        print(f"Hidden dim: {self.hidden_dim}, Sequence length: {self.sequence_length}")
        print(f"Device: {self.device}")
        
        # Import score function for test evaluation
        if test_df is not None:
            from evaluation import score as compute_score
        
        # Prepare features and targets
        features_df = self.prepare_features(df)
        features = features_df.values
        targets = df['market_forward_excess_returns'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create dataset (excluding last sequence_length rows that don't have targets)
        valid_length = len(features) - self.sequence_length
        features_scaled = features_scaled[:valid_length + self.sequence_length]
        targets = targets[self.sequence_length:valid_length + self.sequence_length]
        
        # Train on full data (no split) - use all data for training
        train_dataset = MarketDataset(
            features_scaled,
            targets,
            self.sequence_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = features_scaled.shape[1]
        self.model = MarketPredictionNetwork(input_dim, self.hidden_dim).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Loss and optimizer with cosine annealing
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        best_train_loss = float('inf')
        best_test_score = float('-inf')
        train_losses = []
        test_scores = []
        baseline_score = None
        
        # Compute baseline score once if test provided
        if test_df is not None:
            n_test = len(test_df)
            baseline_positions = [1.0] * n_test  # 100% market
            solution_df = test_df[['date_id', 'forward_returns', 'risk_free_rate']].copy()
            submission_df = pd.DataFrame({'date_id': test_df['date_id'], 'prediction': baseline_positions})
            try:
                baseline_score = compute_score(solution_df, submission_df, 'date_id')
            except Exception:
                baseline_score = None
            if baseline_score is not None:
                print(f"Baseline (100% Market) Test Score: {baseline_score:.4f}")
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False
            )
            for batch_x, batch_y in progress_bar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                returns_pred, uncertainty_pred = self.model(batch_x)
                
                # Loss: MSE for returns + uncertainty regularization
                mse_loss = nn.MSELoss()(returns_pred, batch_y.squeeze())
                uncertainty_loss = uncertainty_pred.mean()  # Penalize high uncertainty
                loss = mse_loss + 0.01 * uncertainty_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            scheduler.step()
            
            # Save best model by training loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(self.model.state_dict(), 'best_market_model.pth')
            
            # Test evaluation every N epochs
            test_score = None
            if test_df is not None and (epoch + 1) % eval_every == 0:
                self.model.eval()
                positions, _, _ = self.predict_positions(test_df)
                # Pad positions if needed
                if len(positions) < len(test_df):
                    pad = [1.0] * (len(test_df) - len(positions))
                    positions = pad + positions
                submission_df = pd.DataFrame({'date_id': test_df['date_id'], 'prediction': positions})
                try:
                    test_score = compute_score(solution_df, submission_df, 'date_id')
                    test_scores.append((epoch + 1, test_score))
                except Exception as e:
                    test_score = None
                    print(f"  [Test eval failed: {e}]")
                
                if test_score is not None and test_score > best_test_score:
                    best_test_score = test_score
                    torch.save(self.model.state_dict(), 'best_market_model_by_score.pth')
            
            # Log
            log_msg = f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}"
            if test_score is not None:
                delta = ""
                if baseline_score is not None:
                    diff = test_score - baseline_score
                    delta = f" (Δ vs baseline: {diff:+.4f})"
                log_msg += f", Test Score: {test_score:.4f}{delta}"
            print(log_msg)
        
        # Load best model (by test score if available, else by train loss)
        if test_df is not None and best_test_score > float('-inf'):
            self.model.load_state_dict(torch.load('best_market_model_by_score.pth'))
            print(f"\n{'='*80}")
            print(f"Training completed! Loaded best model by test score: {best_test_score:.4f}")
            if baseline_score is not None:
                print(f"Improvement over baseline: {best_test_score - baseline_score:+.4f}")
        else:
            self.model.load_state_dict(torch.load('best_market_model.pth'))
            print(f"\n{'='*80}")
            print(f"Training completed! Best train loss: {best_train_loss:.6f}")
        
        return train_losses, [], test_scores
    
    def predict_positions(self, df):
        """Predict betting positions for test data"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        self.model.eval()
        
        # Prepare features
        features_df = self.prepare_features(df)
        features_scaled = self.scaler.transform(features_df.values)
        
        positions = []
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for i in range(self.sequence_length, len(features_scaled)):
                # Get sequence
                seq = features_scaled[i - self.sequence_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                # Predict
                returns_pred, uncertainty_pred = self.model(seq_tensor)
                pred_return = returns_pred.item()
                pred_uncertainty = uncertainty_pred.item()
                
                predictions.append(pred_return)
                uncertainties.append(pred_uncertainty)
                
                # Calculate market volatility
                recent_returns = df['market_forward_excess_returns'].iloc[
                    max(0, i - self.betting_strategy.market_vol_window):i
                ]
                market_vol = recent_returns.std()
                
                # Get risk-free rate
                risk_free = df['risk_free_rate'].iloc[i]
                
                # Calculate position
                position = self.betting_strategy.calculate_position(
                    pred_return, pred_uncertainty, market_vol, risk_free
                )
                positions.append(position)
        
        return positions, predictions, uncertainties


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_strategy(df, positions):
    """Evaluate betting strategy performance"""
    # Pad positions to match df length
    padded_positions = [0.0] * (len(df) - len(positions)) + positions
    
    # Calculate portfolio returns
    portfolio_returns = []
    for i in range(len(df)):
        if i < len(df) - len(positions):
            portfolio_returns.append(0.0)
        else:
            pos = padded_positions[i]
            market_ret = df['market_forward_excess_returns'].iloc[i]
            rf = df['risk_free_rate'].iloc[i]
            
            # Portfolio return = position * market_return + (1-position) * rf_rate
            # But our position is 0-2, so we need to handle leverage
            if pos <= 1:
                port_ret = pos * market_ret + (1 - pos) * rf
            else:
                # Leverage: borrow at rf to invest more
                port_ret = pos * market_ret - (pos - 1) * rf
            
            portfolio_returns.append(port_ret)
    
    portfolio_returns = np.array(portfolio_returns)
    market_returns = df['market_forward_excess_returns'].values
    
    # Calculate metrics
    strategy_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    market_sharpe = market_returns.mean() / market_returns.std() * np.sqrt(252)
    
    strategy_cumret = (1 + portfolio_returns).cumprod()[-1] - 1
    market_cumret = (1 + market_returns).cumprod()[-1] - 1
    
    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE")
    print("=" * 80)
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
    print(f"Market Sharpe Ratio: {market_sharpe:.4f}")
    print(f"Strategy Cumulative Return: {strategy_cumret:.2%}")
    print(f"Market Cumulative Return: {market_cumret:.2%}")
    print(f"Average Position: {np.mean(positions):.4f}")
    print(f"Position Std: {np.std(positions):.4f}")
    print("=" * 80)
    
    return portfolio_returns


# ============================================================================
# COMPLETE PIPELINE WITH EVALUATION
# ============================================================================

def run_complete_pipeline(train_df, test_df, solution_df):
    """
    Complete end-to-end pipeline: train, predict, and evaluate
    
    Args:
        train_df: Training data (imputed)
        test_df: Test data (imputed) 
        solution_df: Test solution with ['date_id', 'forward_returns', 'risk_free_rate']
    
    Returns:
        predictor: Trained model
        positions: Predicted positions
        evaluator: StrategyEvaluator object with all results
    """
    print("\n" + "=" * 100)
    print("COMPLETE MARKET PREDICTION PIPELINE")
    print("=" * 100)
    
    # Step 1: Train the model
    print("\n[1/4] Training prediction model...")
    predictor = MarketPredictor(sequence_length=60, hidden_dim=256)
    train_losses, val_losses = predictor.train(train_df, epochs=100, batch_size=64, lr=0.0001)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as 'training_curves.png'")
    
    # Step 2: Generate predictions
    print("\n[2/4] Generating predictions on test set...")
    positions, predictions, uncertainties = predictor.predict_positions(test_df)
    
    # Pad positions to match test_df length if needed
    if len(positions) < len(test_df):
        # Pad with neutral position (1.0 = 100% market exposure)
        padding = [1.0] * (len(test_df) - len(positions))
        positions = padding + positions
    
    print(f"Generated {len(positions)} position predictions")
    print(f"Position range: [{min(positions):.4f}, {max(positions):.4f}]")
    print(f"Average position: {np.mean(positions):.4f}")
    
    # Step 3: Evaluate against baselines
    print("\n[3/4] Evaluating strategy against baselines...")
    from evaluation import evaluate_with_baselines
    
    evaluator = evaluate_with_baselines(
        solution_df=solution_df,
        your_positions=positions,
        strategy_name='Deep Learning Strategy'
    )
    
    # Step 4: Additional analysis
    print("\n[4/4] Generating additional insights...")
    
    # Plot prediction confidence
    if len(predictions) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Predictions over time
        axes[0].plot(predictions, label='Predicted Returns', alpha=0.7)
        axes[0].fill_between(
            range(len(predictions)), 
            np.array(predictions) - np.array(uncertainties),
            np.array(predictions) + np.array(uncertainties),
            alpha=0.3, label='Uncertainty Band'
        )
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title('Model Predictions with Uncertainty')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Positions over time
        # Remove padding to match predictions length
        positions_to_plot = positions[-len(predictions):] if len(positions) > len(predictions) else positions
        axes[1].plot(positions_to_plot, label='Position', color='green', alpha=0.7)
        axes[1].axhline(y=1.0, color='blue', linestyle='--', label='100% Market', linewidth=1)
        axes[1].axhline(y=1.2, color='orange', linestyle='--', label='120% (Vol Target)', linewidth=1)
        axes[1].fill_between(range(len(positions_to_plot)), 0, positions_to_plot, alpha=0.3, color='green')
        axes[1].set_xlabel('Test Days')
        axes[1].set_ylabel('Position (0-2)')
        axes[1].set_title('Position Allocation Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 2])
        
        plt.tight_layout()
        plt.savefig('predictions_and_positions.png', dpi=300, bbox_inches='tight')
        print("Predictions plot saved as 'predictions_and_positions.png'")
    
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE!")
    print("=" * 100)
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - strategy_comparison.png")
    print("  - predictions_and_positions.png")
    print("  - best_market_model.pth (saved model)")
    
    return predictor, positions, evaluator


# Example usage:
"""
# Complete pipeline from imputed data to evaluation:

# 1. Load and prepare data
train_df = pd.read_csv('train_imputed.csv')
test_df = pd.read_csv('test_imputed.csv')

# 2. Prepare solution DataFrame for evaluation
solution_df = test_df[['date_id', 'forward_returns', 'risk_free_rate']].copy()

# 3. Run complete pipeline
predictor, positions, evaluator = run_complete_pipeline(
    train_df=train_df,
    test_df=test_df,
    solution_df=solution_df
)

# 4. Create submission file
submission = pd.DataFrame({
    'date_id': test_df['date_id'],
    'prediction': positions
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")
"""