import pandas as pd

def solution(df):
    df = df.copy()

    col_ra = df.isnull().mean(axis=0)
    df = df.loc[:, col_ra <= 0.5]

    row_ra = df.isnull().mean(axis=1)
    df = df.loc[row_ra <= 0.5]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df.reset_index(drop=True)