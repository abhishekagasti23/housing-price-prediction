import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


def load_data(filepath):
    """Load housing dataset"""
    return pd.read_csv(filepath)


def handle_missing_values(df):
    """Fill missing numeric values with median"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def encode_categorical(df):
    """One-hot encode categorical columns"""
    if 'ocean_proximity' in df.columns:
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    return df


def split_scale_save(df, target='median_house_value'):
    """Split data, scale features, and save artifacts"""

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save("data/X_train.npy", X_train_scaled)
    np.save("data/X_test.npy", X_test_scaled)
    np.save("data/y_train.npy", y_train.values)
    np.save("data/y_test.npy", y_test.values)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    df = load_data("data/housing.csv")
    df = handle_missing_values(df)
    df = encode_categorical(df)
    split_scale_save(df)
