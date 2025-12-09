# Market Prediction & Imputation Solution 📈

## Overview

This is a complete end-to-end solution for the Hull Tactical Market Prediction Competition, featuring:

1. **Novel Deep Learning Imputation** - Handle missing time series data intelligently
2. **Advanced Market Prediction Model** - Transformer + LSTM architecture
3. **Sophisticated Betting Strategy** - Kelly Criterion with volatility targeting
4. **Comprehensive Evaluation** - Compare against baselines using official metric

---

## 📊 Understanding Your Data

Your dataset has an interesting structure where columns start having data at different time points:

- **Always present**: date_id, D1-D9, forward_returns, risk_free_rate, market_forward_excess_returns
- **Starting later**: E, I, M, P, S, V features (starting from row 1005-6968)

This requires sophisticated imputation before modeling!

---

## 🎯 The Competition Metric

The official metric is a **volatility-adjusted Sharpe ratio** with two penalties:

```
Final Score = Sharpe Ratio / (Volatility Penalty × Return Penalty)
```

### Volatility Penalty
- **Safe Zone**: Volatility ≤ 1.2× Market Volatility → No penalty
- **Penalty Zone**: Volatility > 1.2× Market → Penalty = 1 + Excess Vol

### Return Penalty  
- **Safe Zone**: Strategy Return ≥ Market Return → No penalty
- **Penalty Zone**: Strategy Return < Market → Penalty = 1 + (Gap²/100)

### Key Insight
You want to **maximize Sharpe ratio** while keeping:
- Volatility ≤ 120% of market volatility
- Returns ≥ market returns

---

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn
```

### Complete Pipeline

```python
from workflow import main

# Run everything: imputation → training → prediction → evaluation
results = main(
    train_path='train.csv',
    test_path='test.csv'
)

# Your submission is ready: submission.csv
```

### Step-by-Step Usage

```python
# 1. Imputation
from imputation import DeepImputer, analyze_missing_data

analyze_missing_data(train_df)  # See where data starts

imputer = DeepImputer(feature_cols, window_size=30, hidden_dim=128)
imputer.train(train_df, epochs=50)
train_imputed = imputer.impute(train_df)

# 2. Model Training
from strategy import MarketPredictor

predictor = MarketPredictor(sequence_length=60, hidden_dim=256)
predictor.train(train_imputed, epochs=100)

# 3. Generate Predictions (0-2 range)
positions, predictions, uncertainties = predictor.predict_positions(test_df)

# 4. Evaluate Against Baselines
from evaluation import evaluate_with_baselines

evaluator = evaluate_with_baselines(
    solution_df=test_df[['date_id', 'forward_returns', 'risk_free_rate']],
    your_positions=positions,
    strategy_name='My Strategy'
)

# 5. Create Submission
submission = pd.DataFrame({
    'date_id': test_df['date_id'],
    'prediction': positions
})
submission.to_csv('submission.csv', index=False)
```

---

## 🏗️ Architecture Details

### 1. Imputation Network

**Multi-Task Bidirectional LSTM with Attention**

```
Input (with NaNs) 
    → Masked LSTM Encoder (captures temporal patterns)
    → Temporal Attention (weights important time steps)
    → Feature-Specific Decoders (E, I, M, P, S, V groups)
    → Imputed Values
```

**Why it works:**
- Bidirectional LSTM sees both past and future context
- Attention focuses on relevant time periods
- Group-specific decoders capture feature correlations
- Trained only on observed values (masked loss)

### 2. Market Prediction Network

**Hybrid Architecture: TCN + Transformer + LSTM**

```
Input Features
    → Temporal Convolutions (local patterns, multi-scale)
    → Multi-Head Self-Attention (feature interactions)
    → Bidirectional LSTM (sequential dependencies)
    → Dual Heads:
        - Return Prediction
        - Uncertainty Estimation
```

**Why it works:**
- TCN captures short-term patterns at multiple scales
- Attention discovers complex feature relationships
- LSTM models temporal dynamics
- Uncertainty guides position sizing

### 3. Adaptive Betting Strategy

**Kelly Criterion + Volatility Targeting + Risk Management**

```python
# Core calculation
kelly_position = (expected_return - rf_rate) / variance²
vol_scalar = target_vol / current_vol
position = kelly_position × vol_scalar × confidence_factor

# Risk overlays
if expected_return < rf_rate: position = min(position, 1.0)
if uncertainty > threshold: position *= 0.5
position = clip(position, 0, 2)
```

**Why it works:**
- Kelly Criterion maximizes long-term growth
- Volatility targeting keeps strategy within 120% vol limit
- Confidence adjustment scales down uncertain predictions
- Risk overlays prevent extreme positions

---

## 📈 Baseline Strategies

### Baseline 0: Risk-Free (position = 0)
- **Return**: Risk-free rate (~2%)
- **Volatility**: Near-zero
- **Sharpe**: Usually low
- **Score**: Low (not enough return)

### Baseline 1: 100% Market (position = 1)
- **Return**: Market return (~10%)
- **Volatility**: Market volatility (~15%)
- **Sharpe**: Market Sharpe (~0.4-0.6)
- **Score**: This is what you need to beat!

### Your Goal
Beat Baseline 1 by achieving:
- Higher Sharpe ratio (better return/risk)
- Controlled volatility (≤ 1.2x market)
- Competitive returns (≥ market)

---

## 📊 Interpreting Results

### Understanding the Comparison Table

```
Strategy                        Official Score    Sharpe    Vol Ratio    Vol Penalty    Return Penalty
---------------------------------------------------------------------------------------------------------
Deep Learning Strategy                 0.8234      1.42         1.15           1.00              1.00  ✅
Baseline 1 (100% Market)               0.6541      0.51         1.00           1.00              1.00
Baseline 0 (Risk-Free)                 0.0892      0.12         0.01           1.00              2.45
```

**What to look for:**
- ✅ **Official Score > Baseline 1** → You're beating the market!
- ✅ **Vol Penalty = 1.00** → No volatility penalty (good)
- ✅ **Return Penalty = 1.00** → No return penalty (good)
- ✅ **Sharpe > 1.0** → Excellent risk-adjusted returns

**Red flags:**
- ⚠️ Vol Penalty > 1.05 → Too much volatility
- ⚠️ Return Penalty > 1.05 → Underperforming market
- ⚠️ Score < Baseline 1 → Need improvement

---

## 🎨 Generated Visualizations

### 1. `training_curves.png`
- Training and validation loss over epochs
- Check for overfitting (val loss increasing)

### 2. `imputation_quality.png`
- Original vs imputed values for sample features
- Verify imputation looks reasonable

### 3. `strategy_comparison.png`
- **Cumulative returns**: See your strategy vs baselines over time
- **Score comparison**: Bar chart of official scores
- **Volatility**: Are you within the 120% limit?
- **Position distribution**: How aggressive is your strategy?
- **Rolling Sharpe**: Performance stability over time
- **Penalty breakdown**: Where are you losing points?

### 4. `predictions_and_positions.png`
- Model predictions with uncertainty bands
- Position allocation over time (0-2 range)

### 5. `metric_explanation.png`
- Visual guide to understanding the metric
- Shows penalty zones and optimal regions

---

## 🔧 Hyperparameter Tuning

### Imputation Model

```python
imputer = DeepImputer(
    feature_cols,
    window_size=30,      # ← Try 20-50 (larger = more context)
    hidden_dim=128       # ← Try 64-256 (larger = more capacity)
)
imputer.train(
    df, 
    epochs=50,          # ← Try 30-100
    batch_size=64,      # ← Try 32-128
    lr=0.001            # ← Try 0.0001-0.01
)
```

### Prediction Model

```python
predictor = MarketPredictor(
    sequence_length=60,  # ← Try 30-120 (days to look back)
    hidden_dim=256       # ← Try 128-512 (model capacity)
)
predictor.train(
    df,
    epochs=100,         # ← Try 50-200
    batch_size=64,      # ← Try 32-128
    lr=0.0001           # ← Try 0.00001-0.001
)
```

### Betting Strategy (in `strategy.py`)

```python
betting_strategy = AdaptiveBettingStrategy(
    target_vol=0.12,        # ← 120% of market (DON'T exceed 1.2)
    kelly_fraction=0.25,    # ← Try 0.1-0.5 (lower = more conservative)
    max_leverage=2.0        # ← Fixed by competition rules
)
```

---

## 💡 Tips for Improvement

### 1. Feature Engineering
Add more features in `MarketPredictor.prepare_features()`:
- Momentum: `(price[t] - price[t-20]) / price[t-20]`
- RSI: Relative Strength Index
- Moving average crossovers
- Volume indicators (if available)

### 2. Ensemble Methods
```python
# Train multiple models with different seeds/parameters
positions_1 = predictor_1.predict_positions(test_df)[0]
positions_2 = predictor_2.predict_positions(test_df)[0]
positions_3 = predictor_3.predict_positions(test_df)[0]

# Average predictions
ensemble_positions = [
    (p1 + p2 + p3) / 3 
    for p1, p2, p3 in zip(positions_1, positions_2, positions_3)
]
```

### 3. Advanced Strategies
- **Regime detection**: Different strategies for bull/bear markets
- **Risk parity**: Balance risk across features
- **Dynamic volatility targeting**: Adjust target based on market conditions

### 4. Validation Strategy
```python
# Time series cross-validation
for fold in range(5):
    train_end = len(df) * (fold + 1) // 6
    val_end = len(df) * (fold + 2) // 6
    
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    
    # Train and validate
    predictor.train(train_data)
    score = evaluate(predictor, val_data)
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Low Score Despite Good Sharpe
**Problem**: High volatility or return penalties
**Solution**: 
- Check vol_ratio - should be ≤ 1.2
- Verify returns - should be ≥ market returns
- Reduce `kelly_fraction` to be more conservative

### Issue 2: Model Predicts Constant Positions
**Problem**: Model not learning meaningful patterns
**Solution**:
- Increase model capacity (hidden_dim)
- Train longer (more epochs)
- Add more diverse features
- Check data quality (imputation worked?)

### Issue 3: Imputation Creates Unrealistic Values
**Problem**: Imputed values too extreme or inconsistent
**Solution**:
- Use RobustScaler instead of StandardScaler
- Add constraints in imputation loss
- Increase window_size for more context

### Issue 4: Overfitting in Training
**Problem**: Training loss ↓ but validation loss ↑
**Solution**:
- Add more dropout
- Use weight decay
- Early stopping
- Data augmentation

---

## 📚 File Structure

```
.
├── imputation.py                 # Deep learning imputation
├── strategy.py                   # Prediction model + betting strategy
├── evaluation.py                 # Evaluation framework
├── workflow.py                   # End-to-end pipeline
├── metrics.py                    # Understand the metric
│
├── train.csv                     # Your training data
├── test.csv                      # Your test data
│
├── train_imputed.csv            # Generated: imputed training data
├── test_imputed.csv             # Generated: imputed test data
├── best_market_model.pth        # Generated: trained model
├── submission.csv               # Generated: final submission
│
└── *.png                        # Generated: visualizations
```

---

## 🎯 Success Checklist

- [ ] Data loaded and analyzed
- [ ] Missing data patterns understood
- [ ] Imputation completed (no NaN remaining)
- [ ] Model trained (val loss converged)
- [ ] Predictions generated (0-2 range validated)
- [ ] **Official Score > Baseline 1** ✅
- [ ] Vol Penalty = 1.00 (no volatility penalty)
- [ ] Return Penalty = 1.00 (no return penalty)
- [ ] Submission file created and validated

---

## 🤝 Contributing & Customization

This is a framework - customize it to your needs:

1. **Different architectures**: Try GRU, Attention-only, etc.
2. **Alternative imputation**: Try MICE, matrix factorization
3. **New features**: Add your domain knowledge
4. **Ensemble strategies**: Combine multiple approaches
5. **Risk management**: Add custom constraints

---

## 📖 References

- **Kelly Criterion**: Optimal bet sizing for maximizing growth
- **Transformer Architecture**: Attention mechanisms for time series
- **Temporal Convolutions**: Multi-scale pattern extraction
- **Portfolio Theory**: Modern portfolio optimization

---

## 🏆 Final Tips

1. **Start simple**: Get baseline working first
2. **Iterate quickly**: Fast experiments → fast learning
3. **Trust the metric**: The official score tells the truth
4. **Volatility matters**: Don't chase returns at the cost of vol
5. **Market benchmark is tough**: Beating it consistently is hard!

Good luck! 🚀

---

*Questions? Check the code comments - they're detailed!*