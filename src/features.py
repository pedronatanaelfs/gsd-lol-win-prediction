"""
Feature engineering utilities.
"""

import pandas as pd


def create_engineered_features(df):
    """
    Create derived features from raw data.
    
    All features are computed from early-game data only (no leakage).
    Creates advantage features (Blue - Red deltas) for various game metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with blue and red team features
        
    Returns
    -------
    pd.DataFrame
        Dataset with engineered features added
    """
    df_eng = df.copy()
    
    # Gold advantage (verify if exists, create if not)
    if 'blueGoldDiff' not in df_eng.columns:
        df_eng['blueGoldDiff'] = df_eng['blueTotalGold'] - df_eng['redTotalGold']
    
    # Experience advantage (verify if exists, create if not)
    if 'blueExperienceDiff' not in df_eng.columns:
        df_eng['blueExperienceDiff'] = df_eng['blueTotalExperience'] - df_eng['redTotalExperience']
    
    # Combat advantages
    df_eng['blueKillAdvantage'] = df_eng['blueKills'] - df_eng['redKills']
    df_eng['blueDeathAdvantage'] = df_eng['redDeaths'] - df_eng['blueDeaths']  # Fewer deaths = advantage
    df_eng['blueAssistAdvantage'] = df_eng['blueAssists'] - df_eng['redAssists']
    
    # Objective advantages
    df_eng['blueDragonAdvantage'] = df_eng['blueDragons'] - df_eng['redDragons']
    df_eng['blueHeraldAdvantage'] = df_eng['blueHeralds'] - df_eng['redHeralds']
    df_eng['blueTowerAdvantage'] = df_eng['blueTowersDestroyed'] - df_eng['redTowersDestroyed']
    df_eng['blueEliteMonsterAdvantage'] = df_eng['blueEliteMonsters'] - df_eng['redEliteMonsters']
    
    # Vision advantages
    df_eng['blueWardAdvantage'] = df_eng['blueWardsPlaced'] - df_eng['redWardsPlaced']
    df_eng['blueWardDestroyedAdvantage'] = df_eng['blueWardsDestroyed'] - df_eng['redWardsDestroyed']
    
    # Farming advantages
    df_eng['blueCSAdvantage'] = df_eng['blueTotalMinionsKilled'] - df_eng['redTotalMinionsKilled']
    df_eng['blueJungleCSAdvantage'] = df_eng['blueTotalJungleMinionsKilled'] - df_eng['redTotalJungleMinionsKilled']
    
    # Level advantage
    df_eng['blueLevelAdvantage'] = df_eng['blueAvgLevel'] - df_eng['redAvgLevel']
    
    return df_eng
