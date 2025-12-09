"""
UNDERSTANDING THE COMPETITION METRIC
====================================

This script helps you understand how the official metric works and
what factors contribute to a good score.

The metric is a volatility-adjusted Sharpe ratio with two penalties:
1. Volatility Penalty: Penalizes strategies with vol > 1.2x market vol
2. Return Penalty: Penalizes strategies that underperform the market
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def visualize_metric_components():
    """
    Create visualizations explaining the metric components
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Understanding the Competition Metric', fontsize=16, fontweight='bold')
    
    # ========================================================================
    # 1. Sharpe Ratio Calculation
    # ========================================================================
    ax1 = axes[0, 0]
    
    # Simulate different return/volatility combinations
    returns = np.linspace(-0.05, 0.15, 50)
    vols = np.linspace(0.05, 0.30, 50)
    
    R, V = np.meshgrid(returns, vols)
    sharpe = R / V * np.sqrt(252)
    
    contour = ax1.contourf(R * 100, V * 100, sharpe, levels=20, cmap='RdYlGn')
    ax1.contour(R * 100, V * 100, sharpe, levels=[0, 0.5, 1.0, 1.5, 2.0], 
                colors='black', linewidths=1, alpha=0.3)
    
    # Mark some interesting points
    ax1.scatter([10], [15], s=200, c='blue', marker='*', 
               edgecolors='black', linewidth=2, label='Target: High Sharpe', zorder=5)
    ax1.scatter([5], [25], s=200, c='red', marker='X', 
               edgecolors='black', linewidth=2, label='Bad: Low Sharpe', zorder=5)
    
    ax1.set_xlabel('Annualized Return (%)', fontsize=11)
    ax1.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax1.set_title('1. Sharpe Ratio = Return / Volatility', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
    
    # ========================================================================
    # 2. Volatility Penalty
    # ========================================================================
    ax2 = axes[0, 1]
    
    vol_ratios = np.linspace(0.5, 2.5, 100)
    vol_penalties = 1 + np.maximum(0, vol_ratios - 1.2)
    
    ax2.plot(vol_ratios, vol_penalties, linewidth=3, color='darkred')
    ax2.fill_between(vol_ratios, 1, vol_penalties, alpha=0.3, color='red', 
                     label='Penalty Region')
    
    # Mark key points
    ax2.axvline(x=1.2, color='orange', linestyle='--', linewidth=2, 
               label='Penalty Threshold (1.2x)')
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1, 
               label='No Penalty')
    
    # Annotate regions
    ax2.text(0.7, 1.5, 'Safe Zone\n(No Penalty)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(1.8, 2.2, 'Penalty Zone\n(Vol Too High)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax2.set_xlabel('Strategy Vol / Market Vol', fontsize=11)
    ax2.set_ylabel('Volatility Penalty Factor', fontsize=11)
    ax2.set_title('2. Volatility Penalty (Keep Vol ≤ 1.2x Market)', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.5, 2.5])
    ax2.set_ylim([0.9, 2.5])
    
    # ========================================================================
    # 3. Return Penalty
    # ========================================================================
    ax3 = axes[1, 0]
    
    return_gaps = np.linspace(-50, 50, 100)  # Annualized return gap (%)
    return_penalties = 1 + (np.maximum(0, return_gaps) ** 2) / 100
    
    ax3.plot(return_gaps, return_penalties, linewidth=3, color='darkblue')
    ax3.fill_between(return_gaps, 1, return_penalties, 
                     where=(return_gaps >= 0), alpha=0.3, color='blue',
                     label='Penalty Region')
    
    # Mark key points
    ax3.axvline(x=0, color='orange', linestyle='--', linewidth=2, 
               label='Break-even (Match Market)')
    ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=1,
               label='No Penalty')
    
    # Annotate regions
    ax3.text(-30, 1.5, 'Beat Market\n(No Penalty)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax3.text(30, 5, 'Underperform\n(Penalty)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax3.set_xlabel('Market Return - Strategy Return (% per year)', fontsize=11)
    ax3.set_ylabel('Return Penalty Factor', fontsize=11)
    ax3.set_title('3. Return Penalty (Beat or Match Market)', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-50, 50])
    ax3.set_ylim([0.9, 10])
    
    # ========================================================================
    # 4. Final Score Calculation
    # ========================================================================
    ax4 = axes[1, 1]
    
    # Show the formula as a flowchart
    ax4.axis('off')
    
    # Title
    ax4.text(0.5, 0.95, 'Final Score Calculation', ha='center', fontsize=14,
            fontweight='bold', transform=ax4.transAxes)
    
    # Step 1: Sharpe Ratio
    rect1 = Rectangle((0.15, 0.75), 0.7, 0.12, linewidth=2, 
                     edgecolor='blue', facecolor='lightblue', alpha=0.5,
                     transform=ax4.transAxes)
    ax4.add_patch(rect1)
    ax4.text(0.5, 0.81, 'Step 1: Calculate Sharpe Ratio', ha='center',
            fontsize=11, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.77, 'Sharpe = (Return - RiskFree) / Vol × √252', ha='center',
            fontsize=9, transform=ax4.transAxes)
    
    # Arrow
    ax4.annotate('', xy=(0.5, 0.73), xytext=(0.5, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2),
                transform=ax4.transAxes)
    
    # Step 2: Calculate Penalties
    rect2 = Rectangle((0.05, 0.50), 0.4, 0.18, linewidth=2, 
                     edgecolor='red', facecolor='lightcoral', alpha=0.5,
                     transform=ax4.transAxes)
    ax4.add_patch(rect2)
    ax4.text(0.25, 0.64, 'Volatility Penalty', ha='center',
            fontsize=10, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.25, 0.60, 'If Vol > 1.2× Market:', ha='center',
            fontsize=8, transform=ax4.transAxes)
    ax4.text(0.25, 0.56, 'Penalty = 1 + Excess Vol', ha='center',
            fontsize=8, transform=ax4.transAxes)
    ax4.text(0.25, 0.52, 'Else: Penalty = 1', ha='center',
            fontsize=8, transform=ax4.transAxes)
    
    rect3 = Rectangle((0.55, 0.50), 0.4, 0.18, linewidth=2, 
                     edgecolor='red', facecolor='lightcoral', alpha=0.5,
                     transform=ax4.transAxes)
    ax4.add_patch(rect3)
    ax4.text(0.75, 0.64, 'Return Penalty', ha='center',
            fontsize=10, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.75, 0.60, 'If Return < Market:', ha='center',
            fontsize=8, transform=ax4.transAxes)
    ax4.text(0.75, 0.56, 'Gap = Market - Strategy', ha='center',
            fontsize=8, transform=ax4.transAxes)
    ax4.text(0.75, 0.52, 'Penalty = 1 + Gap² / 100', ha='center',
            fontsize=8, transform=ax4.transAxes)
    
    # Arrows
    ax4.annotate('', xy=(0.25, 0.45), xytext=(0.25, 0.50),
                arrowprops=dict(arrowstyle='->', lw=2),
                transform=ax4.transAxes)
    ax4.annotate('', xy=(0.75, 0.45), xytext=(0.75, 0.50),
                arrowprops=dict(arrowstyle='->', lw=2),
                transform=ax4.transAxes)
    
    # Step 3: Final Score
    rect4 = Rectangle((0.15, 0.25), 0.7, 0.15, linewidth=3, 
                     edgecolor='green', facecolor='lightgreen', alpha=0.5,
                     transform=ax4.transAxes)
    ax4.add_patch(rect4)
    ax4.text(0.5, 0.36, 'Final Score', ha='center',
            fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.31, 'Score = Sharpe / (Vol Penalty × Return Penalty)', ha='center',
            fontsize=10, transform=ax4.transAxes)
    ax4.text(0.5, 0.27, 'Capped at 1,000,000', ha='center',
            fontsize=9, style='italic', transform=ax4.transAxes)
    
    # Arrows converging
    ax4.annotate('', xy=(0.4, 0.40), xytext=(0.25, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2),
                transform=ax4.transAxes)
    ax4.annotate('', xy=(0.6, 0.40), xytext=(0.75, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2),
                transform=ax4.transAxes)
    
    # Key Insight Box
    insight_box = Rectangle((0.05, 0.05), 0.9, 0.15, linewidth=2, 
                           edgecolor='purple', facecolor='lavender', alpha=0.7,
                           transform=ax4.transAxes)
    ax4.add_patch(insight_box)
    ax4.text(0.5, 0.16, '🎯 Key Insight', ha='center',
            fontsize=11, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.11, 'Maximize Sharpe ratio while keeping volatility ≤ 1.2x market', ha='center',
            fontsize=9, transform=ax4.transAxes)
    ax4.text(0.5, 0.07, 'and returns ≥ market returns', ha='center',
            fontsize=9, transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig


def simulate_score_scenarios():
    """
    Simulate various scenarios to understand score dynamics
    """
    print("\n" + "=" * 100)
    print("SCORE SCENARIOS ANALYSIS")
    print("=" * 100)
    
    # Market baseline
    market_return = 0.10  # 10% annual
    market_vol = 0.15     # 15% annual
    market_sharpe = market_return / market_vol * np.sqrt(252)
    
    scenarios = [
        {
            'name': 'Baseline 0: Risk-Free',
            'return': 0.02,  # 2% risk-free
            'vol': 0.001,    # Near-zero volatility
        },
        {
            'name': 'Baseline 1: 100% Market',
            'return': market_return,
            'vol': market_vol,
        },
        {
            'name': 'Good: Higher Return, Same Vol',
            'return': 0.12,
            'vol': market_vol,
        },
        {
            'name': 'Good: Same Return, Lower Vol',
            'return': market_return,
            'vol': 0.12,
        },
        {
            'name': 'Bad: High Vol (1.5x)',
            'return': 0.12,
            'vol': market_vol * 1.5,
        },
        {
            'name': 'Bad: Lower Return',
            'return': 0.08,
            'vol': market_vol,
        },
        {
            'name': 'Optimal: High Return, Controlled Vol',
            'return': 0.14,
            'vol': market_vol * 1.1,
        },
    ]
    
    print(f"\nMarket Benchmark: {market_return*100:.1f}% return, {market_vol*100:.1f}% vol, "
          f"Sharpe = {market_sharpe:.2f}")
    print("\n" + "-" * 100)
    
    results = []
    for scenario in scenarios:
        ret = scenario['return']
        vol = scenario['vol']
        
        # Calculate Sharpe
        sharpe = ret / vol * np.sqrt(252)
        
        # Calculate penalties
        vol_ratio = vol / market_vol
        excess_vol = max(0, vol_ratio - 1.2)
        vol_penalty = 1 + excess_vol
        
        return_gap = max(0, (market_return - ret) * 100 * 252)
        return_penalty = 1 + (return_gap ** 2) / 100
        
        # Final score
        score = sharpe / (vol_penalty * return_penalty)
        
        results.append({
            'Scenario': scenario['name'],
            'Return': f"{ret*100:.1f}%",
            'Vol': f"{vol*100:.1f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Vol Penalty': f"{vol_penalty:.2f}",
            'Ret Penalty': f"{return_penalty:.2f}",
            'Final Score': f"{score:.2f}"
        })
        
        print(f"\n{scenario['name']}")
        print(f"  Return: {ret*100:.1f}%, Vol: {vol*100:.1f}%, Sharpe: {sharpe:.2f}")
        print(f"  Vol Penalty: {vol_penalty:.2f} (vol ratio: {vol_ratio:.2f})")
        print(f"  Return Penalty: {return_penalty:.2f} (gap: {return_gap:.1f})")
        print(f"  ➜ Final Score: {score:.2f}")
    
    print("\n" + "-" * 100)
    print("\nSUMMARY TABLE:")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("KEY TAKEAWAYS:")
    print("=" * 100)
    print("1. High Sharpe ratio is the foundation - maximize return/risk")
    print("2. Keep volatility below 1.2x market to avoid penalties")
    print("3. Match or beat market returns to avoid return penalties")
    print("4. The optimal strategy: High returns + controlled volatility + low drawdowns")
    print("=" * 100)


def main():
    """
    Run all metric explanations
    """
    # Create visualizations
    print("Generating metric visualization...")
    fig = visualize_metric_components()
    plt.savefig('metric_explanation.png', dpi=300, bbox_inches='tight')
    print("Saved: metric_explanation.png")
    plt.show()
    
    # Simulate scenarios
    simulate_score_scenarios()


if __name__ == "__main__":
    main()