import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import seaborn as sns

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

def save_cleaned(df):
    df.to_csv("df_clean.csv", index = False)

def save_dataset(df, file_path):
    df.to_csv(file_path, index = False)
    print(f"Saved cleaned dataset to {file_path}")

# Step 4: Feature engineering
    
# Before aggregation - daily variation - compute from daily data
    
def pivot_raw(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['time']).dt.normalize()

    raw_pivot = df.pivot_table(
        index=['id', 'time', 'date'],
        columns='variable',
        values='value'
    ).reset_index()
    
    raw_pivot.columns.name = None
    return raw_pivot


def daily_variation(raw_pivot):
    numeric_columns = raw_pivot.select_dtypes(include="number").columns
    
    scaler = MinMaxScaler()
    normalized = raw_pivot.copy()
    normalized[numeric_columns] = scaler.fit_transform(raw_pivot[numeric_columns])
    
    new_cols = {}
    for var in numeric_columns:
        new_cols[f"{var}_std"] = normalized.groupby(['id', 'date'])[var].transform('std')
    
    variation = pd.DataFrame(new_cols, index=raw_pivot.index)
    variation['id'] = raw_pivot['id']
    variation['date'] = raw_pivot['date']
    return variation.groupby(['id', 'date']).first().reset_index()


def aggregate_daily(raw_pivot):
    df = raw_pivot.copy()
    numeric_columns = df.select_dtypes(include="number").columns
    
    agg_dict = {var: ("sum" if var in USAGE_VARS else "mean") for var in numeric_columns}
    
    daily = df.groupby(['id', 'date']).agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    return daily

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

# Missing values

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

def interpolate(df):
    df[STATE_VARS] = df[STATE_VARS].groupby(df["id"]).apply(lambda x: x.interpolate())

def knn_impute(df):
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df[STATE_VARS] = imputer.fit_transform(df[STATE_VARS])
    return df

def save_cleaned(df, name):
    df.to_csv(name, index = False)
    print(f"Saved cleaned data to {name}")

# Feature engineering 

# Sliding windows     
def sliding_window_features(df, windows=[3, 7, 14, 30]):
    df = df.sort_values(['id', 'date'])
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != 'id']
    
    new_cols = {}
    for var in numeric_cols:
        for w in windows:
            mean_col = f"{var}_mean_{w}d"
            new_cols[mean_col] = (df.groupby('id')[var]
                                    .transform(lambda x: x.rolling(w, min_periods=1).mean()))
            new_cols[f"{var}_diff_{w}d"] = df.groupby('id')[mean_col].diff() if mean_col in df.columns else new_cols[mean_col].groupby(df['id']).diff()
    
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


### CLASSIFICATION ###

# We implement two types of algorithms
# 1. Instance based algorithm: not a sequential, temporal prediction; makes a prediction indepdenetly based on the previous row (id,date)
#    - The temporal nature is encoded in the features, e.g. sliding window features, daily variation, etc. (shows importance of feature engineering)
# 2. Temporal model: utilizes the temporal nature of the data 
#    - Learns also the sequential relationships 

## 1. Instance based classification: Random Forest

## 2. Temporal classification: Recurrent neural network

def create_sequences(df, window_size=7):
    # Creates sequences of size of window, with target - next day 

    X = []
    y = [] 

    feature_columns = [c for c in df.columns if c not in ('id', 'date', 'mood_class', "mood_class_target")]

    for uid, user_df in df.groupby('id'):
        user_df = user_df.sort_values("date")
        features = user_df[feature_columns].values
        targets = user_df['mood_class_target'].values

        for i in range(len(user_df) - window_size):
            X.append(features[i:i+window_size])
            y.append(targets[i+window_size])

    return np.array(X, dtype = np.float32), np.array(y)

# Dataset class for PyTorch 

class MoodDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class MoodLSTM(nn.Module):
    def __init__(self, n_features, n_classes=4, hidden_size=64):
        super().__init__() # from parent class nn.Module
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))
    

def main():
    ### DATA CLEANING: TASK 1 ###
    # Load data:

    df = load_data(file_path = "dataset_mood_smartphone.csv")

    # Outliers:
    df = drop_index_column(df)
    df = to_datetime(df)
    df = remove_negatives(df)
    df = remove_outliers(df)

    # Save cleaned dataset before aggregation
    save_cleaned(df, "data_cleaned_before_agg.csv")

    # Daily standard deviation 
    raw_pivot = pivot_raw(df)
    std_daily = daily_variation(raw_pivot)
    # Result - NaNs here mean that either there were no entries, or only one, so the standard deviation doesn't exist
    print(std_daily.head(5))
    #save_dataset(std_daily, "daily_standard_variation.csv")

    # Feature engineering
    df = aggregate_daily(raw_pivot)
    df = remove_impossible_daily(df)
    df = reindex_all_users(df)

    # Missing values
    df = fill_usage_vars(df)
    df = trim_all_users(df)
    df = knn_impute(df)

    # Final column order
    cols = ['id', 'date'] + [c for c in df.columns if c not in ('id', 'date')]
    df = df[cols]

    print(df.head(10))

    # Save final cleaned dataset
    save_cleaned(df, "data_cleaned.csv")

    # Feature engineering 

    df = sliding_window_features(df)  

    cols = ['id', 'date'] + [c for c in df.columns if c not in ('id', 'date')]

    print("Final dataset with features + sliding windows:")
    print(df.head())

    print("Investigate mood variable (normalized):")
    print(df['mood'].describe())
    print(df['mood'].head(20))

    df = df.merge(std_daily, on=['id', 'date'], how='left')
    std_cols = [c for c in df.columns if c.endswith('_std')]
    df[std_cols] = df[std_cols].fillna(0)

    # Engineer classes for supervised learning: mood class
    # Mood is an average mood during the day 
    df['mood_class'] = pd.cut(df['mood'], bins=[-np.inf, 4,6,8, np.inf], labels=['low', 'medium', 'high', 'very_high'])
    # We use up to 4 for low because there are no values below 2 to have a class "very_low"
    print(df['mood_class'].value_counts())
    print(df['mood_class'].head(10))


    # Check missing values?
    print(df.isnull().sum()[df.isnull().sum() > 0])
    # Result: 27 NaN values in every variable - in the first row for each user diff() has nothing to subtract, so produces this
    # We drop this row
    diff_cols = [c for c in df.columns if '_diff_' in c]
    df = df.dropna(subset=diff_cols)


    save_dataset(df, "data_with_features.csv")

    ### CLASSIFICATION: TASK 2 ###

    # 1. Instance based: Random Forest 
    # 1. Train test split
    class_order = ['low', 'medium', 'high', 'very_high']
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    X = df.drop(columns=['id', 'date', 'mood', 'mood_class'])
    y = df['mood_class'].map(class_to_idx)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Fit Random Forest Classifier

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Make a prediction 

    y_pred = rf.predict(X_test)

    # Evaluate performance 

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(y_test, y_pred, target_names=class_order, output_dict=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels = class_order, ax = axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")

    report_df = pd.DataFrame(class_report).T
    report_df = report_df.drop(columns='support')  
    sns.heatmap(report_df.iloc[:-2], annot=True, fmt=".2f", cmap="Blues", 
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Classification Report")

    plt.suptitle(f"Random Forest — Accuracy: {accuracy:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig("rf_evaluation.png")
    plt.show()

    # Feature importance 
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10_indices = indices[:10]
    feature_names = np.array(df.drop(columns=['id', 'date', 'mood_class']).columns)

    plt.figure(figsize=(12, 6))
    plt.title("Top 10 Feature Importances")
    plt.bar(range(10), importances[top_10_indices], color="r", align="center")
    plt.xticks(range(10), [feature_names[i] for i in top_10_indices], rotation=90)
    plt.xlim([-1, 10])
    plt.tight_layout()
    plt.savefig("feature_importance.png", bbox_inches='tight')
    plt.show()

    # Visualize first 3 decision trees 

    for decision in range(3):
        tree = rf.estimators_[decision]
        plt.figure(figsize=(20,10))
        plot_tree(tree, filled = True, feature_names=X.columns, class_names=class_order, rounded=True, 
              max_depth=3)
        plt.savefig(f"tree_{decision}.png")
        plt.close()

    # 2. Temporal model: RNN 
    
    # We need to prepare the sequential data 
    # Because the target is the next day's mood, we shift the mood class column by -1 (for supervised learning)
        
    df['mood_class_target'] = df.groupby('id')['mood_class'].shift(-1)
    # We drop the last row for each user because it has no target
    df = df.dropna(subset=['mood_class_target'])

    X, y = create_sequences(df, window_size=7)

    # Train test split (sequentially)
    threshold = int(0.8 * len(X))
    X_train, X_test = X[:threshold], X[threshold:]
    y_train, y_test = y[:threshold], y[threshold:]

    # Scale 
    scaler = StandardScaler()
    n_samples, window, n_features = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(-1, window, n_features)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(-1, window, n_features)

    # Change string classes into integers
    classes = ['low', 'medium', 'high', 'very_high']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([class_to_idx[c] for c in y_train])
    y_test  = np.array([class_to_idx[c] for c in y_test])

    train_dataset = MoodDataset(X_train, y_train)
    test_dataset = MoodDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, shuffle = False, batch_size=32)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    model = MoodLSTM(n_features = n_features, n_classes=4, hidden_size = 64)

    # Train - with Adam Optimizer 
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train() # this turns the dropout on, the next lines train 
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step() # update weights 
        print(f"Epoch {epoch+1}/20  loss: {loss.item():.4f}")

    # Evaluation 
    
    model.eval()

if __name__ == "__main__":
    main()
