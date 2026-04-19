import pandas as pd

def load_data(file_path, dtype_dict=None):
    df = pd.read_csv(file_path, dtype = dtype_dict, low_memory=False)
    return df

def create_next_day_target(df):
    df = df.sort_values(['id', 'date']).copy()

    # target = next day mood per user
    df['target_mood'] = df.groupby('id')['mood'].shift(-1)

    # remove last day per user (no target)
    df = df.dropna(subset=['target_mood'])

    return df

def prepare_ml_data(df):
    df = create_next_day_target(df)

    # features (remove target + raw future label)
    X = df.drop(columns=['target_mood', 'date'])

    # keep id separately if needed for temporal model
    ids = X['id'].values

    # encode id (important for tree models)
    X_encoded = pd.get_dummies(X, columns=['id'])

    y = df['target_mood']

    return X_encoded, y, ids, df

def time_split(df, test_ratio=0.2):
    df = df.sort_values(['id', 'date']).copy()

    split_data = []

    for uid, user_df in df.groupby('id'):
        cut = int(len(user_df) * (1 - test_ratio))

        train = user_df.iloc[:cut]
        test = user_df.iloc[cut:]

        split_data.append((train, test))

    train_df = pd.concat([t[0] for t in split_data])
    test_df = pd.concat([t[1] for t in split_data])

    return train_df, test_df

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_non_temporal(train_df, test_df):

    X_train, y_train, _, _ = prepare_ml_data(train_df)
    X_test, y_test, _, _ = prepare_ml_data(test_df)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    print("Non-temporal Random Forest MAE:", mae)

    return model

df = load_data('data_cleaned.csv')
train_df, test_df = time_split(df)

# NON-TEMPORAL
rf_model = train_non_temporal(train_df, test_df)

