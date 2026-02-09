"""
Model evaluation metrics and utilities.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies.
    Lower values indicate better calibration.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for probability calibration
        
    Returns
    -------
    float
        Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        # Include lower bound for first bin (to catch predictions exactly at 0.0)
        if bin_lower == 0:
            in_bin = (y_pred >= bin_lower) & (y_pred <= bin_upper)
        else:
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_metrics(y_true, y_pred_proba, y_pred=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    y_pred : array-like, optional
        Predicted binary labels (if None, uses 0.5 threshold)
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }
    
    if y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        })
    
    return metrics


def calculate_calibration_metrics(y_true, y_pred_proba):
    """
    Calculate calibration-specific metrics.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
        
    Returns
    -------
    dict
        Dictionary of calibration metrics
    """
    return {
        'ece': expected_calibration_error(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
