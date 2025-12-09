"""
COMPLETE WORKFLOW: FROM RAW DATA TO SUBMISSION
==============================================

This script provides a complete end-to-end pipeline for:
1. Analyzing missing data
2. Imputing missing values using deep learning
3. Training market prediction model
4. Generating predictions
5. Evaluating against baselines
6. Creating submission file

Author: Your Name
Date: 2025
"""

import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import your modules (assuming they're in separate files)
# from imputation import DeepImputer, analyze_missing_data
# from strategy import MarketPredictor, run_complete_pipeline
# from evaluation import StrategyEvaluator, evaluate_with_baselines


# ============================================================================
# STEP 1: DATA LOADING AND MISSING DATA ANALYSIS
# ============================================================================

def load_and_analyze_data(train_path, test_path=None):
    """
    Load data and analyze missing patterns
    """
    print("=" * 100)
    print("STEP 1: DATA LOADING AND ANALYSIS")
    print("=" * 100)
    
    # Load training data
    print(f"\nLoading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Training data shape: {train_df.shape}")
    
    # Analyze missing data
    print("\nAnalyzing missing data patterns...")
    from imputation import analyze_missing_data
    missing_analysis = analyze_missing_data(train_df)
    
    # Load test data if provided
    test_df = None
    if test_path:
        print(f"\nLoading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df, missing_analysis


# ============================================================================
# STEP 2: IMPUTATION
# ============================================================================

def impute_missing_values(train_df, test_df=None, imputer_window=30, imputer_hidden=128, 
                          imputer_epochs=5, imputer_batch_size=64, imputer_lr=0.001):
    """
    Impute missing values using deep learning
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        imputer_window: Window size for imputer (default: 30)
        imputer_hidden: Hidden dimension for imputer (default: 128)
        imputer_epochs: Training epochs for imputer (default: 5)
        imputer_batch_size: Batch size for imputer (default: 64)
        imputer_lr: Learning rate for imputer (default: 0.001)
    """
    print("\n" + "=" * 100)
    print("STEP 2: DEEP LEARNING IMPUTATION")
    print("=" * 100)
    
    from imputation import DeepImputer
    
    # Identify feature columns
    feature_cols = [col for col in train_df.columns if col not in 
                   ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]
    
    print(f"\nFeature columns to impute: {len(feature_cols)}")
    print(f"Imputer config: window={imputer_window}, hidden={imputer_hidden}, epochs={imputer_epochs}")
    
    # Initialize and train imputer on training data
    print("\nTraining imputation model...")
    imputer = DeepImputer(feature_cols, window_size=imputer_window, hidden_dim=imputer_hidden)
    imputer.train(train_df, epochs=imputer_epochs, batch_size=imputer_batch_size, lr=imputer_lr)
    
    # Impute training data
    print("\nImputing training data...")
    train_imputed = imputer.impute(train_df)
    
    # Impute test data if provided
    test_imputed = None
    if test_df is not None:
        print("\nImputing test data...")
        test_imputed = imputer.impute(test_df)
    
    # Save imputed data
    train_imputed.to_csv('train_imputed.csv', index=False)
    print("\nSaved: train_imputed.csv")
    
    if test_imputed is not None:
        test_imputed.to_csv('test_imputed.csv', index=False)
        print("Saved: test_imputed.csv")
    
    return train_imputed, test_imputed, imputer


# ============================================================================
# STEP 3: TRAIN MARKET PREDICTION MODEL
# ============================================================================

def train_prediction_model(train_imputed, test_imputed=None, eval_every=10,
                           seq_length=60, hidden_dim=512, epochs=100, 
                           batch_size=64, lr=0.0001):
    """
    Train the market prediction model on full training data.
    Evaluate on test every `eval_every` epochs.
    
    Args:
        train_imputed: Imputed training dataframe
        test_imputed: Test dataframe for periodic evaluation
        eval_every: Evaluate on test every N epochs
        seq_length: Sequence length for model (default: 60)
        hidden_dim: Hidden dimension for model (default: 512)
        epochs: Training epochs (default: 100)
        batch_size: Batch size (default: 64)
        lr: Learning rate (default: 0.0001)
    """
    print("\n" + "=" * 100)
    print("STEP 3: TRAINING MARKET PREDICTION MODEL")
    print("=" * 100)
    
    from strategy import MarketPredictor
    
    print(f"Training on full dataset: {len(train_imputed)} rows")
    print(f"Model config: seq_length={seq_length}, hidden_dim={hidden_dim}, epochs={epochs}, lr={lr}")
    if test_imputed is not None:
        print(f"Evaluating on test set every {eval_every} epochs")
    
    # Initialize predictor with configured model
    predictor = MarketPredictor(sequence_length=seq_length, hidden_dim=hidden_dim)
    
    # Train model on full data with periodic test evaluation
    train_losses, val_losses, test_scores = predictor.train(
        train_imputed, 
        epochs=epochs, 
        batch_size=batch_size, 
        lr=lr,
        test_df=test_imputed,
        eval_every=eval_every
    )
    
    print("\nModel training complete!")
    print("Model saved as: best_market_model.pth")
    
    return predictor, train_losses, val_losses


# ============================================================================
# STEP 4: GENERATE PREDICTIONS
# ============================================================================

def generate_predictions(predictor, test_imputed):
    """
    Generate position predictions for test data
    """
    print("\n" + "=" * 100)
    print("STEP 4: GENERATING PREDICTIONS")
    print("=" * 100)
    
    print("\nGenerating predictions...")
    positions, predictions, uncertainties = predictor.predict_positions(test_imputed)
    
    # Pad if necessary
    if len(positions) < len(test_imputed):
        padding = [1.0] * (len(test_imputed) - len(positions))
        positions = padding + positions
        print(f"\nPadded {len(padding)} positions with neutral allocation (1.0)")
    
    print(f"\nGenerated {len(positions)} predictions")
    print(f"Position statistics:")
    print(f"  Mean: {np.mean(positions):.4f}")
    print(f"  Std:  {np.std(positions):.4f}")
    print(f"  Min:  {np.min(positions):.4f}")
    print(f"  Max:  {np.max(positions):.4f}")
    
    # Show position distribution
    print(f"\nPosition distribution:")
    print(f"  0.0-0.5 (Conservative): {sum(1 for p in positions if p < 0.5)}")
    print(f"  0.5-1.0 (Moderate):     {sum(1 for p in positions if 0.5 <= p < 1.0)}")
    print(f"  1.0-1.5 (Aggressive):   {sum(1 for p in positions if 1.0 <= p < 1.5)}")
    print(f"  1.5-2.0 (Leveraged):    {sum(1 for p in positions if p >= 1.5)}")
    
    return positions, predictions, uncertainties


# ============================================================================
# STEP 5: EVALUATE AGAINST BASELINES
# ============================================================================

def evaluate_strategy(test_imputed, positions):
    """
    Evaluate strategy against baselines using official metric
    """
    print("\n" + "=" * 100)
    print("STEP 5: STRATEGY EVALUATION")
    print("=" * 100)
    
    from evaluation import evaluate_with_baselines
    
    # Prepare solution DataFrame
    solution_df = test_imputed[['date_id', 'forward_returns', 'risk_free_rate']].copy()
    
    # Evaluate with baselines
    evaluator = evaluate_with_baselines(
        solution_df=solution_df,
        your_positions=positions,
        strategy_name='Deep Learning Strategy'
    )
    
    return evaluator


# ============================================================================
# STEP 6: CREATE SUBMISSION FILE
# ============================================================================

def create_submission(test_imputed, positions, output_path='submission.csv'):
    """
    Create final submission file
    """
    print("\n" + "=" * 100)
    print("STEP 6: CREATING SUBMISSION FILE")
    print("=" * 100)
    
    submission = pd.DataFrame({
        'date_id': test_imputed['date_id'],
        'prediction': positions
    })
    
    # Validate submission
    assert len(submission) == len(test_imputed), "Submission length mismatch!"
    assert submission['prediction'].min() >= 0, "Positions below 0!"
    assert submission['prediction'].max() <= 2, "Positions above 2!"
    assert not submission['prediction'].isna().any(), "NaN in predictions!"
    
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved: {output_path}")
    print(f"   Rows: {len(submission)}")
    print(f"   Columns: {list(submission.columns)}")
    
    return submission


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(train_path='train.csv', test_path='test.csv', 
         skip_imputation=False, skip_training=False,
         # Imputer params
         imputer_window=30, imputer_hidden=128, imputer_epochs=5,
         imputer_batch_size=64, imputer_lr=0.001,
         # Strategy params
         seq_length=60, hidden_dim=512, epochs=100,
         batch_size=64, lr=0.0001, eval_every=10):
    """
    Complete end-to-end pipeline
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        skip_imputation: If True, load pre-imputed data
        skip_training: If True, load pre-trained model
        imputer_window: Window size for imputer
        imputer_hidden: Hidden dimension for imputer
        imputer_epochs: Training epochs for imputer
        imputer_batch_size: Batch size for imputer
        imputer_lr: Learning rate for imputer
        seq_length: Sequence length for prediction model
        hidden_dim: Hidden dimension for prediction model
        epochs: Training epochs for prediction model
        batch_size: Batch size for prediction model
        lr: Learning rate for prediction model
        eval_every: Evaluate on test every N epochs
    """
    print("\n" + "=" * 100)
    print("COMPLETE MARKET PREDICTION PIPELINE")
    print("=" * 100)
    print(f"\nTrain data: {train_path}")
    print(f"Test data:  {test_path}")
    print(f"Skip imputation: {skip_imputation}")
    print(f"Skip training:   {skip_training}")
    
    # Step 1: Load data
    if not skip_imputation:
        train_df, test_df, missing_analysis = load_and_analyze_data(train_path, test_path)
        
        # Step 2: Impute
        train_imputed, test_imputed, imputer = impute_missing_values(
            train_df, test_df,
            imputer_window=imputer_window,
            imputer_hidden=imputer_hidden,
            imputer_epochs=imputer_epochs,
            imputer_batch_size=imputer_batch_size,
            imputer_lr=imputer_lr
        )
    else:
        print("\nLoading pre-imputed data...")
        train_imputed = pd.read_csv('train_imputed.csv')
        test_imputed = pd.read_csv('test_imputed.csv')
        print(f"Train imputed shape: {train_imputed.shape}")
        print(f"Test imputed shape:  {test_imputed.shape}")
    
    # Step 3: Train model
    if not skip_training:
        predictor, train_losses, val_losses = train_prediction_model(
            train_imputed, test_imputed,
            eval_every=eval_every,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
    else:
        print("\nLoading pre-trained model...")
        from strategy import MarketPredictor
        predictor = MarketPredictor(sequence_length=seq_length, hidden_dim=hidden_dim)
        import torch
        predictor.model = torch.load('best_market_model.pth')
        print("Model loaded successfully!")
    
    # Step 4: Generate predictions
    positions, predictions, uncertainties = generate_predictions(predictor, test_imputed)
    
    # Step 5: Evaluate
    evaluator = evaluate_strategy(test_imputed, positions)
    
    # Step 6: Create submission
    submission = create_submission(test_imputed, positions)
    
    # Summary
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE!")
    print("=" * 100)
    print("\n📁 Generated Files:")
    print("   - train_imputed.csv (imputed training data)")
    print("   - test_imputed.csv (imputed test data)")
    print("   - best_market_model.pth (trained model)")
    print("   - training_curves.png (training visualization)")
    print("   - strategy_comparison.png (evaluation results)")
    print("   - predictions_and_positions.png (predictions over time)")
    print("   - submission.csv (final submission)")
    
    print("\n📊 Key Results:")
    results = evaluator.compare_all_strategies()
    best_score = results[results['Strategy'] == 'Deep Learning Strategy']['Official Score'].values[0]
    baseline_1 = results[results['Strategy'] == 'Baseline 1 (100% Market)']['Official Score'].values[0]
    
    print(f"   Your Strategy Score:    {best_score:.4f}")
    print(f"   Baseline 1 Score:       {baseline_1:.4f}")
    
    if best_score > baseline_1:
        improvement = ((best_score - baseline_1) / abs(baseline_1)) * 100
        print(f"   🎉 BEATS BASELINE by {improvement:.2f}%!")
    else:
        decline = ((baseline_1 - best_score) / abs(baseline_1)) * 100
        print(f"   ⚠️  Underperforms by {decline:.2f}%")
    
    print("\n✅ Ready for submission: submission.csv")
    print("=" * 100)
    
    return {
        'predictor': predictor,
        'evaluator': evaluator,
        'positions': positions,
        'submission': submission
    }


# ============================================================================
# QUICK USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Hull Tactical market prediction workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument("--train-path", default="train.csv", help="Path to the training CSV file")
    parser.add_argument("--test-path", default="test.csv", help="Path to the test CSV file")
    
    # Skip flags
    parser.add_argument(
        "--skip-imputation",
        action="store_true",
        help="Use existing train_imputed.csv/test_imputed.csv instead of recomputing",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load best_market_model.pth instead of retraining the predictor",
    )
    
    # Imputer parameters
    parser.add_argument("--imputer-window", type=int, default=30, help="Window size for imputer")
    parser.add_argument("--imputer-hidden", type=int, default=128, help="Hidden dimension for imputer")
    parser.add_argument("--imputer-epochs", type=int, default=5, help="Training epochs for imputer")
    parser.add_argument("--imputer-batch-size", type=int, default=64, help="Batch size for imputer")
    parser.add_argument("--imputer-lr", type=float, default=0.001, help="Learning rate for imputer")
    
    # Strategy model parameters
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length for prediction model")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for prediction model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for prediction model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for prediction model")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for prediction model")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate on test every N epochs")
    
    args = parser.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        skip_imputation=args.skip_imputation,
        skip_training=args.skip_training,
        # Imputer params
        imputer_window=args.imputer_window,
        imputer_hidden=args.imputer_hidden,
        imputer_epochs=args.imputer_epochs,
        imputer_batch_size=args.imputer_batch_size,
        imputer_lr=args.imputer_lr,
        # Strategy params
        seq_length=args.seq_length,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
    )


"""
CUSTOMIZATION TIPS
==================

1. Adjust Imputation Settings:
   - window_size: Larger = more context (try 20-50)
   - hidden_dim: Larger = more capacity (try 64-256)
   - epochs: More = better fit (try 30-100)

2. Adjust Prediction Model:
   - sequence_length: How many days to look back (try 30-120)
   - hidden_dim: Model capacity (try 128-512)
   - epochs: Training duration (try 50-200)

3. Adjust Betting Strategy (in strategy.py):
   - target_vol: Target volatility (default 0.12 = 120%)
   - kelly_fraction: How aggressive (try 0.1-0.5)
   - max_leverage: Maximum position (default 2.0)

4. Feature Engineering:
   Add more features in MarketPredictor.prepare_features():
   - Momentum indicators
   - Volatility measures
   - Technical indicators
   - Cross-sectional features

5. Ensemble Methods:
   Train multiple models and average their predictions:
   
   positions_1 = predictor_1.predict_positions(test_df)[0]
   positions_2 = predictor_2.predict_positions(test_df)[0]
   ensemble_positions = [(p1 + p2) / 2 for p1, p2 in zip(positions_1, positions_2)]
"""