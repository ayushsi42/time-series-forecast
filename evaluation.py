import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas.api.types

# ============================================================================
# OFFICIAL COMPETITION METRIC
# ============================================================================

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).
    
    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.
    """
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution.copy()
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)


# ============================================================================
# COMPREHENSIVE STRATEGY EVALUATOR
# ============================================================================

class StrategyEvaluator:
    """
    Comprehensive evaluation framework for comparing strategies against baselines
    """
    def __init__(self, solution_df: pd.DataFrame, row_id_column: str = 'date_id'):
        """
        Args:
            solution_df: DataFrame with columns ['date_id', 'forward_returns', 'risk_free_rate']
            row_id_column: Name of the row ID column
        """
        self.solution_df = solution_df.copy()
        self.row_id_column = row_id_column
        self.results = {}
        
    def evaluate_strategy(self, positions: List[float], strategy_name: str) -> Dict:
        """
        Evaluate a single strategy and return detailed metrics
        
        Args:
            positions: List of positions (0-2) for each day
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with all performance metrics
        """
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            self.row_id_column: self.solution_df[self.row_id_column],
            'prediction': positions
        })
        
        # Calculate official score
        try:
            official_score = score(self.solution_df, submission_df, self.row_id_column)
        except ParticipantVisibleError as e:
            print(f"Error evaluating {strategy_name}: {e}")
            return None
        
        # Calculate detailed metrics
        solution = self.solution_df.copy()
        solution['position'] = positions
        solution['strategy_returns'] = (
            solution['risk_free_rate'] * (1 - solution['position']) + 
            solution['position'] * solution['forward_returns']
        )
        
        # Excess returns
        strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
        market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
        
        # Cumulative returns
        strategy_cumulative = (1 + solution['strategy_returns']).cumprod()
        market_cumulative = (1 + solution['forward_returns']).cumprod()
        
        # Annualized metrics
        trading_days_per_yr = 252
        n_days = len(solution)
        years = n_days / trading_days_per_yr
        
        strategy_total_return = strategy_cumulative.iloc[-1] - 1
        market_total_return = market_cumulative.iloc[-1] - 1
        
        strategy_annualized_return = (1 + strategy_total_return) ** (1 / years) - 1
        market_annualized_return = (1 + market_total_return) ** (1 / years) - 1
        
        # Volatility
        strategy_vol = solution['strategy_returns'].std() * np.sqrt(trading_days_per_yr)
        market_vol = solution['forward_returns'].std() * np.sqrt(trading_days_per_yr)
        
        # Sharpe ratio
        strategy_sharpe = (
            strategy_excess_returns.mean() / solution['strategy_returns'].std() * 
            np.sqrt(trading_days_per_yr)
        )
        market_sharpe = (
            market_excess_returns.mean() / solution['forward_returns'].std() * 
            np.sqrt(trading_days_per_yr)
        )
        
        # Drawdown analysis
        strategy_drawdown = self._calculate_drawdown(strategy_cumulative)
        market_drawdown = self._calculate_drawdown(market_cumulative)
        
        # Win rate
        win_rate = (solution['strategy_returns'] > solution['risk_free_rate']).mean()
        
        # Calculate penalties from official metric
        excess_vol = max(0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0
        vol_penalty = 1 + excess_vol
        
        strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
        strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / n_days) - 1
        market_excess_cumulative = (1 + market_excess_returns).prod()
        market_mean_excess_return = (market_excess_cumulative) ** (1 / n_days) - 1
        
        return_gap = max(
            0,
            (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
        )
        return_penalty = 1 + (return_gap**2) / 100
        
        metrics = {
            'strategy_name': strategy_name,
            'official_score': official_score,
            'total_return': strategy_total_return,
            'annualized_return': strategy_annualized_return,
            'volatility': strategy_vol,
            'sharpe_ratio': strategy_sharpe,
            'max_drawdown': strategy_drawdown['max_drawdown'],
            'win_rate': win_rate,
            'avg_position': np.mean(positions),
            'position_std': np.std(positions),
            'vol_ratio': strategy_vol / market_vol if market_vol > 0 else 0,
            'vol_penalty': vol_penalty,
            'return_penalty': return_penalty,
            'excess_vol': excess_vol,
            'return_gap': return_gap,
            'cumulative_series': strategy_cumulative,
            'returns_series': solution['strategy_returns'],
            'positions': positions
        }
        
        self.results[strategy_name] = metrics
        return metrics
    
    def _calculate_drawdown(self, cumulative_returns):
        """Calculate drawdown statistics"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return {
            'max_drawdown': drawdown.min(),
            'drawdown_series': drawdown
        }
    
    def create_baselines(self):
        """
        Create baseline strategies:
        - Baseline 0: All risk-free (position = 0)
        - Baseline 1: All market (position = 1)
        """
        n = len(self.solution_df)
        
        # Baseline 0: Risk-free
        baseline_0 = [0.0] * n
        self.evaluate_strategy(baseline_0, 'Baseline 0 (Risk-Free)')
        
        # Baseline 1: 100% Market
        baseline_1 = [1.0] * n
        self.evaluate_strategy(baseline_1, 'Baseline 1 (100% Market)')
        
    def compare_all_strategies(self) -> pd.DataFrame:
        """
        Create comparison table of all evaluated strategies
        """
        if not self.results:
            print("No strategies evaluated yet!")
            return None
        
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Strategy': name,
                'Official Score': metrics['official_score'],
                'Total Return': metrics['total_return'],
                'Ann. Return': metrics['annualized_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate'],
                'Avg Position': metrics['avg_position'],
                'Vol Ratio': metrics['vol_ratio'],
                'Vol Penalty': metrics['vol_penalty'],
                'Return Penalty': metrics['return_penalty']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Official Score', ascending=False)
        return df
    
    def plot_comprehensive_analysis(self, figsize=(20, 12)):
        """
        Create comprehensive visualization comparing all strategies
        """
        if not self.results:
            print("No strategies evaluated yet!")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        for name, metrics in self.results.items():
            ax1.plot(metrics['cumulative_series'].values, label=name, linewidth=2)
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Official Score Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        strategies = list(self.results.keys())
        scores = [self.results[s]['official_score'] for s in strategies]
        colors = ['red' if 'Baseline' in s else 'green' for s in strategies]
        ax2.barh(strategies, scores, color=colors, alpha=0.7)
        ax2.set_xlabel('Official Score')
        ax2.set_title('Official Score Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Sharpe Ratio Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        sharpes = [self.results[s]['sharpe_ratio'] for s in strategies]
        ax3.barh(strategies, sharpes, color=colors, alpha=0.7)
        ax3.set_xlabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Volatility Comparison
        ax4 = fig.add_subplot(gs[1, 2])
        vols = [self.results[s]['volatility'] * 100 for s in strategies]
        ax4.barh(strategies, vols, color=colors, alpha=0.7)
        ax4.set_xlabel('Volatility (%)')
        ax4.set_title('Volatility Comparison', fontweight='bold')
        ax4.axvline(x=self.results['Baseline 1 (100% Market)']['volatility'] * 100 * 1.2, 
                   color='orange', linestyle='--', label='1.2x Market Vol')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Position Distribution (for non-baseline strategies)
        ax5 = fig.add_subplot(gs[2, 0])
        for name, metrics in self.results.items():
            if 'Baseline' not in name:
                ax5.hist(metrics['positions'], bins=50, alpha=0.5, label=name)
        ax5.set_xlabel('Position (0-2)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Position Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Rolling Sharpe Ratio
        ax6 = fig.add_subplot(gs[2, 1])
        window = 60
        for name, metrics in self.results.items():
            returns = metrics['returns_series']
            rolling_sharpe = (
                returns.rolling(window).mean() / returns.rolling(window).std() * 
                np.sqrt(252)
            )
            ax6.plot(rolling_sharpe.values, label=name, alpha=0.7)
        ax6.set_xlabel('Trading Days')
        ax6.set_ylabel('Rolling Sharpe (60-day)')
        ax6.set_title('Rolling Sharpe Ratio', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 7. Penalty Analysis
        ax7 = fig.add_subplot(gs[2, 2])
        vol_penalties = [self.results[s]['vol_penalty'] for s in strategies]
        return_penalties = [self.results[s]['return_penalty'] for s in strategies]
        x = np.arange(len(strategies))
        width = 0.35
        ax7.bar(x - width/2, vol_penalties, width, label='Vol Penalty', alpha=0.7)
        ax7.bar(x + width/2, return_penalties, width, label='Return Penalty', alpha=0.7)
        ax7.set_xticks(x)
        ax7.set_xticklabels(strategies, rotation=45, ha='right')
        ax7.set_ylabel('Penalty Factor')
        ax7.set_title('Penalty Breakdown', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.axhline(y=1, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        return fig
    
    def print_detailed_comparison(self):
        """Print detailed text comparison"""
        print("\n" + "=" * 100)
        print("DETAILED STRATEGY COMPARISON")
        print("=" * 100)
        
        comparison_df = self.compare_all_strategies()
        
        # Format the dataframe for better display
        pd.options.display.float_format = '{:.4f}'.format
        print(comparison_df.to_string(index=False))
        
        print("\n" + "=" * 100)
        print("KEY INSIGHTS")
        print("=" * 100)
        
        # Find best strategy
        best_strategy = comparison_df.iloc[0]['Strategy']
        best_score = comparison_df.iloc[0]['Official Score']
        
        # Compare against baselines
        baseline_1_score = self.results['Baseline 1 (100% Market)']['official_score']
        baseline_0_score = self.results['Baseline 0 (Risk-Free)']['official_score']
        
        print(f"\n🏆 BEST STRATEGY: {best_strategy}")
        print(f"   Official Score: {best_score:.4f}")
        
        if best_strategy != 'Baseline 1 (100% Market)':
            improvement = ((best_score - baseline_1_score) / abs(baseline_1_score)) * 100
            print(f"\n📈 IMPROVEMENT OVER BASELINE 1 (100% Market):")
            print(f"   Score Improvement: {improvement:+.2f}%")
            print(f"   Baseline 1 Score: {baseline_1_score:.4f}")
            
            if improvement > 0:
                print(f"   ✅ Your strategy BEATS the market benchmark!")
            else:
                print(f"   ❌ Your strategy underperforms the market benchmark")
        
        if best_strategy != 'Baseline 0 (Risk-Free)':
            improvement_0 = ((best_score - baseline_0_score) / abs(baseline_0_score)) * 100
            print(f"\n📊 IMPROVEMENT OVER BASELINE 0 (Risk-Free):")
            print(f"   Score Improvement: {improvement_0:+.2f}%")
            print(f"   Baseline 0 Score: {baseline_0_score:.4f}")
        
        # Analyze penalties
        for name in self.results:
            if name == best_strategy and 'Baseline' not in name:
                metrics = self.results[name]
                print(f"\n🔍 PENALTY ANALYSIS FOR {name}:")
                print(f"   Volatility Ratio: {metrics['vol_ratio']:.4f} (target: ≤1.20)")
                print(f"   Volatility Penalty: {metrics['vol_penalty']:.4f} (optimal: 1.00)")
                print(f"   Return Penalty: {metrics['return_penalty']:.4f} (optimal: 1.00)")
                
                if metrics['vol_penalty'] > 1.01:
                    print(f"   ⚠️  Strategy is being penalized for excess volatility!")
                if metrics['return_penalty'] > 1.01:
                    print(f"   ⚠️  Strategy is being penalized for underperforming market!")
        
        print("\n" + "=" * 100)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def evaluate_with_baselines(solution_df: pd.DataFrame, 
                            your_positions: List[float],
                            strategy_name: str = 'Your Strategy'):
    """
    Complete evaluation pipeline with baselines
    
    Args:
        solution_df: DataFrame with ['date_id', 'forward_returns', 'risk_free_rate']
        your_positions: Your predicted positions (0-2) for each day
        strategy_name: Name for your strategy
    """
    # Initialize evaluator
    evaluator = StrategyEvaluator(solution_df)
    
    # Create baselines
    print("Creating baseline strategies...")
    evaluator.create_baselines()
    
    # Evaluate your strategy
    print(f"\nEvaluating {strategy_name}...")
    evaluator.evaluate_strategy(your_positions, strategy_name)
    
    # Print detailed comparison
    evaluator.print_detailed_comparison()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = evaluator.plot_comprehensive_analysis()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization as 'strategy_comparison.png'")
    
    return evaluator


# Example usage:
"""
# Assuming you have:
# - solution_df with columns: ['date_id', 'forward_returns', 'risk_free_rate']
# - your_positions: list of predictions from your model

evaluator = evaluate_with_baselines(
    solution_df=test_df,
    your_positions=your_model_positions,
    strategy_name='Deep Learning Strategy'
)

# You can also evaluate additional strategies
evaluator.evaluate_strategy(another_positions, 'Alternative Strategy')
evaluator.plot_comprehensive_analysis()
"""