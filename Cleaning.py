import pandas as pd
import numpy as np

### load ###################
df = pd.read_csv("dataset_mood_smartphone.csv", index_col=0)
df['time'] = pd.to_datetime(df['time'])

df_clean = df.copy()
print(f"Loaded: {len(df_clean)} rows, {df_clean['id'].nunique()} users")


### remove impossible values ###########################
before = len(df_clean)

# All appCat and screen duration values should be non-negative
is_duration = df_clean['variable'].str.startswith('appCat') | (df_clean['variable'] == 'screen')
df_clean = df_clean[~(is_duration & (df_clean['value'] < 0))]
print(f"Removed {before - len(df_clean)} rows with negative duration values")

# From investigating value ranges per variable, we see that activity, mood,
# circumplex.arousal and circumplex.valence are all inside their defined ranges.
# Therefore there are no impossible values there to remove.

df_clean['date'] = df_clean['time'].dt.date


### convert to daily level ###########################
# # Variables that accumulate over the day are summed
# (screen, all appCat.*, call, sms)
SUM_VARS = (
    set(df_clean.loc[df_clean['variable'].str.startswith('appCat'), 'variable'].unique())
    | {'screen', 'call', 'sms'}
)

# Variables that represent a state we take the mean
# (mood, circumplex.arousal, circumplex.valence, activity)

mask_sum  = df_clean['variable'].isin(SUM_VARS)

daily_sum  = (df_clean[mask_sum]
              .groupby(['id', 'date', 'variable'])['value']
              .sum()
              .reset_index())

daily_mean = (df_clean[~mask_sum]
              .groupby(['id', 'date', 'variable'])['value']
              .mean()
              .reset_index())

daily = pd.concat([daily_sum, daily_mean], ignore_index=True)


### one row per user × day, one column per variable #####################
daily_wide = daily.pivot_table(
    index=['id', 'date'],
    columns='variable',
    values='value'
).reset_index()

daily_wide.columns.name = None
daily_wide['date'] = pd.to_datetime(daily_wide['date'])


### check for impossible values arter summing #######################
# screen time and any appCat duration cannot exceed 86400 s (24 h).
SECONDS_IN_DAY = 86_400
duration_cols = [c for c in daily_wide.columns if c.startswith('appCat') or c == 'screen']

impossible_mask = daily_wide[duration_cols] > SECONDS_IN_DAY
n_impossible = impossible_mask.values.sum()
daily_wide[duration_cols] = daily_wide[duration_cols].mask(impossible_mask)
print(f"Set {n_impossible} post-aggregation impossible duration values to NaN (> 24 h)")


### We insert rows for every missing day within each user's personal date range ######

def reindex_user(user_df):
    user_df = user_df.sort_values('date').set_index('date')

    # Create full daily date range for this user
    full_range = pd.date_range(user_df.index.min(), user_df.index.max(), freq='D')

    # Reindex to insert missing days
    user_df = user_df.reindex(full_range)
    user_df.index.name = 'date'

    return user_df

parts = []

for uid, udf in daily_wide.groupby('id'):
    udf = udf.drop(columns='id')  # remove id before reindexing
    result = reindex_user(udf)
    result['id'] = uid           # add id back after reindex
    parts.append(result.reset_index())

daily_final = pd.concat(parts, ignore_index=True)


### fill NaN with 0 ##########################
# For call, sms, screen, and all appCat.* NaN means no activity so we fill with 0
ZERO_FILL_VARS = [c for c in daily_final.columns
                  if c in ('call', 'sms', 'screen') or c.startswith('appCat')]

daily_final[ZERO_FILL_VARS] = daily_final[ZERO_FILL_VARS].fillna(0)


# Put id and date first for readability
cols = ['id', 'date'] + [c for c in daily_final.columns if c not in ('id', 'date')]
daily_final = daily_final[cols]


### save ###############
output_path = "data_cleaned.csv"
daily_final.to_csv(output_path, index=False)

print(f"\nFinal dataset: {daily_final.shape[0]} rows × {daily_final.shape[1]} columns")
print(f"Users: {daily_final['id'].nunique()}, Date range: {daily_final['date'].min().date()} → {daily_final['date'].max().date()}")
print(f"Saved to: {output_path}")