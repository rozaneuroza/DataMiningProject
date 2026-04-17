import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

USAGE_VARS = ['call', 'sms', 'screen', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
STATE_VARS = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]

SECONDS_IN_DAY = 86400

# Task 1: EDA, Data Cleaning & Feature Engineering 

# Step 1: Load data

def load_data(file_path, dtype_dict=None):
    df = pd.read_csv(file_path, dtype = dtype_dict, low_memory=False)
    return df

def drop_index_column(df):
    df = df[df.columns[1:]]
    return df 

def to_datetime(df):
    df['time'] = pd.to_datetime(df['time'])
    return df


# Step 2: Remove negative values and outliers/impossible values 

def remove_negatives(df):
    is_duration = df['variable'].str.startswith('appCat') | (df['variable'] == 'screen')
    df = df[~(is_duration & (df['value'] < 0))]
    return df 

def remove_outliers(df, low=0.01, high=0.99):

    results = []
    variables = df['variable'].unique()

    for var in variables:
        subset = df[df["variable"] == var].copy()

        # Known ranges
        if var == "mood":
            subset = subset[subset["value"].between(1, 10)]

        elif var == "circumplex.arousal":
            subset = subset[subset["value"].between(-2, 2)]

        elif var == "circumplex.valence":
            subset = subset[subset["value"].between(-2, 2)]

        elif var == "activity":
            subset = subset[subset["value"].between(0, 1)]

        else:
            lower = subset["value"].quantile(low)
            upper = subset["value"].quantile(high)
            subset = subset[subset["value"].between(lower, upper)]

        results.append(subset)

    df = pd.concat(results, ignore_index=True)
    return df



# Step 4: Feature engineering

# Aggregate daily values
    
def aggregate_values_daily(df):
    df['date'] = df['time'].dt.date   # IMPORTANT: separate date column

    mask_sum = df['variable'].isin(USAGE_VARS)

    daily_sum = (df[mask_sum]
                 .groupby(['id', 'date', 'variable'])['value']
                 .sum()
                 .reset_index())

    daily_mean = (df[~mask_sum]
                  .groupby(['id', 'date', 'variable'])['value']
                  .mean()
                  .reset_index())

    daily = pd.concat([daily_sum, daily_mean], ignore_index=True)

    daily_pivot = daily.pivot_table(
        index=['id', 'date'],
        columns='variable',
        values='value'
    ).reset_index()

    daily_pivot.columns.name = None
    daily_pivot['date'] = pd.to_datetime(daily_pivot['date'])

    return daily_pivot

def remove_impossible_daily(df):
    duration_cols = [c for c in df.columns if c.startswith('appCat') or c == 'screen']

    impossible_mask = df[duration_cols] > SECONDS_IN_DAY
    df[duration_cols] = df[duration_cols].mask(impossible_mask)

    return df

def reindex_user(user_df):
    user_df = user_df.sort_values('date').set_index('date')

    full_range = pd.date_range(user_df.index.min(), user_df.index.max(), freq='D')
    user_df = user_df.reindex(full_range)
    user_df.index.name = 'date'

    return user_df


def reindex_all_users(df):
    parts = []

    for uid, udf in df.groupby('id'):
        udf = udf.drop(columns='id')
        result = reindex_user(udf)
        result['id'] = uid
        parts.append(result.reset_index())

    return pd.concat(parts, ignore_index=True)


##### Missing values

def fill_usage_vars(df):
    usage_cols = [c for c in df.columns if c in USAGE_VARS]
    df[usage_cols] = df[usage_cols].fillna(0)
    return df


def trim_before_first_state(user_df):
    user_df = user_df.sort_values('date').reset_index(drop=True)
    has_state = user_df[STATE_VARS].notna().any(axis=1)
    if not has_state.any():
        return pd.DataFrame(columns=user_df.columns)
    first_valid_pos = has_state.idxmax()
    return user_df.loc[first_valid_pos:].reset_index(drop=True)  # ← fix here


def trim_all_users(df):
    return (
        df.groupby('id', group_keys=False)
        .apply(trim_before_first_state)
        .reset_index(drop=True)
    )

def knn_impute(df):
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df[STATE_VARS] = imputer.fit_transform(df[STATE_VARS])
    return df

def save_cleaned(df, name):
    df.to_csv(name, index = False)
    print(f"Saved cleaned data to {name}")


def main():
    # Load data:

    df = load_data(file_path = "dataset_mood_smartphone.csv")

    # Outliers:
    df = drop_index_column(df)
    df = to_datetime(df)
    df = remove_negatives(df)
    df = remove_outliers(df)

    # Save cleaned dataset before aggregation
    save_cleaned(df, "data_cleaned_before_agg.csv")

    # Feature engineering
    df = aggregate_values_daily(df)
    df = remove_impossible_daily(df)
    df = reindex_all_users(df)

    # Missing values
    df = fill_usage_vars(df)
    df = trim_all_users(df)
    df = knn_impute(df)

    # Final column order
    cols = ['id', 'date'] + [c for c in df.columns if c not in ('id', 'date')]
    df = df[cols]

    # Save final cleaned dataset
    save_cleaned(df, "data_cleaned.csv")

if __name__ == "__main__":
    main()
