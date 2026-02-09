"""
Data loading and validation utilities.
"""

import pandas as pd
from pathlib import Path


def load_data(data_path):
    """
    Load the League of Legends dataset.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    return df


def validate_data(df, target='blueWins', game_id='gameId'):
    """
    Validate dataset integrity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate
    target : str
        Name of target column
    game_id : str
        Name of game ID column
        
    Returns
    -------
    dict
        Validation results
    """
    results = {
        'missing_values': df.isnull().sum().sum(),
        'duplicate_gameids': df[game_id].duplicated().sum(),
        'target_distribution': df[target].value_counts(normalize=True).to_dict(),
        'shape': df.shape
    }
    return results


def prepare_target_features(df, target='blueWins', exclude_cols=None):
    """
    Separate target and feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    target : str
        Name of target column
    exclude_cols : list, optional
        Additional columns to exclude from features
        
    Returns
    -------
    tuple
        (X, y) where X is features and y is target
    """
    if exclude_cols is None:
        exclude_cols = ['gameId']
    else:
        exclude_cols = exclude_cols.copy()
    
    if target not in exclude_cols:
        exclude_cols.append(target)
    
    y = df[target].copy()
    X = df.drop(columns=exclude_cols).copy()
    
    return X, y
