"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def set_plot_style(style='whitegrid', figsize=(12, 6)):
    """
    Set matplotlib and seaborn plotting style.
    
    Parameters
    ----------
    style : str
        Seaborn style name
    figsize : tuple
        Default figure size (width, height)
    """
    sns.set_style(style)
    plt.rcParams['figure.figsize'] = figsize


def plot_feature_importance(feature_names, importances, top_n=15, figsize=(10, 8)):
    """
    Plot feature importance bar chart.
    
    Parameters
    ----------
    feature_names : array-like
        Feature names
    importances : array-like
        Feature importance values
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Sort by importance
    indices = np.argsort(importances)[-top_n:][::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    return fig


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, ax=None):
    """
    Plot calibration curve (reliability diagram).
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    from sklearn.calibration import calibration_curve
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax
