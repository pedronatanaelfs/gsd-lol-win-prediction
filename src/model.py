"""
Model training and evaluation utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def create_logistic_regression(random_state=42, max_iter=1000, solver='lbfgs'):
    """
    Create a Logistic Regression model.
    
    Parameters
    ----------
    random_state : int
        Random seed
    max_iter : int
        Maximum iterations
    solver : str
        Solver algorithm
        
    Returns
    -------
    LogisticRegression
        Configured model
    """
    return LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver=solver
    )


def create_hist_gradient_boosting(
    random_state=42,
    max_iter=100,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-7
):
    """
    Create a Histogram-based Gradient Boosting model.
    
    Parameters
    ----------
    random_state : int
        Random seed
    max_iter : int
        Maximum iterations
    learning_rate : float
        Learning rate
    max_depth : int
        Maximum tree depth
    min_samples_leaf : int
        Minimum samples per leaf
    early_stopping : bool
        Enable early stopping
    validation_fraction : float
        Fraction of data for validation
    n_iter_no_change : int
        Iterations without improvement before stopping
    tol : float
        Tolerance for early stopping
        
    Returns
    -------
    HistGradientBoostingClassifier
        Configured model
    """
    return HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol
    )


def create_calibrated_model(base_model, method='sigmoid', cv=5, n_jobs=-1):
    """
    Create a calibrated classifier.
    
    Parameters
    ----------
    base_model : sklearn estimator
        Base model to calibrate
    method : str
        Calibration method ('sigmoid' for Platt scaling, 'isotonic' for isotonic regression)
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    CalibratedClassifierCV
        Calibrated model
    """
    return CalibratedClassifierCV(
        base_model,
        method=method,
        cv=cv,
        n_jobs=n_jobs
    )


def prepare_scaler():
    """
    Create a StandardScaler instance.
    
    Returns
    -------
    StandardScaler
        Scaler instance
    """
    return StandardScaler()
