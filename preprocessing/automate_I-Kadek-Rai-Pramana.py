"""
Automated Preprocessing Script - Heart Disease Dataset
=======================================================
Script untuk melakukan preprocessing data Heart Disease secara otomatis.
Menghasilkan data yang siap digunakan untuk pelatihan model.

Author: I Kadek Rai Pramana
Version: 2.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load raw Heart Disease dataset."""
    if filepath is None:
        # Cari file di beberapa lokasi
        possible_paths = [
            'heart_disease_raw.csv',
            '../heart_disease_raw.csv',
            '../../heart_disease_raw.csv',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break

    if filepath is None or not os.path.exists(filepath):
        # Download dari UCI jika tidak ada
        print("[INFO] Downloading Heart Disease dataset from UCI...")
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=cols, na_values='?')
    else:
        df = pd.read_csv(filepath)

    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data: handle missing values dan duplikat."""
    print(f"\n[STEP 1] Cleaning Data...")
    initial_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"  - Duplicates removed: {initial_rows - len(df)}")

    # Handle missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  - Missing values found: {missing[missing > 0].to_dict()}")
        # Fill numeric columns with median
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled '{col}' with median: {median_val}")
    else:
        print(f"  - No missing values found")

    print(f"  - Rows after cleaning: {len(df)}")
    return df


def remove_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Hapus outlier menggunakan metode IQR pada kolom numerik."""
    print(f"\n[STEP 2] Removing Outliers (IQR method, factor={factor})...")
    initial_rows = len(df)

    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        if removed > 0:
            print(f"  - '{col}': removed {removed} outliers (range: {lower:.1f} - {upper:.1f})")

    print(f"  - Total outliers removed: {initial_rows - len(df)}")
    print(f"  - Rows after outlier removal: {len(df)}")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi target multi-class menjadi binary classification."""
    print(f"\n[STEP 3] Creating Binary Target...")
    # Original target: 0 = no disease, 1-4 = disease
    # Binary: 0 = no disease, 1 = has disease
    df['heart_disease'] = (df['target'] >= 1).astype(int)
    df = df.drop('target', axis=1)

    print(f"  - Class distribution:")
    print(f"    No Disease (0): {(df['heart_disease'] == 0).sum()}")
    print(f"    Has Disease (1): {(df['heart_disease'] == 1).sum()}")
    return df


def scale_and_split(df: pd.DataFrame, test_size: float = 0.2,
                    output_dir: str = 'heart_disease_preprocessing') -> dict:
    """Lakukan scaling dan split data."""
    print(f"\n[STEP 4] Scaling & Splitting Data...")

    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']

    # Train-test split dengan stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  - Train: {X_train.shape[0]} samples")
    print(f"  - Test: {X_test.shape[0]} samples")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns, index=X_test.index
    )

    # Gabungkan kembali dengan target
    train_df = X_train_scaled.copy()
    train_df['heart_disease'] = y_train.values
    test_df = X_test_scaled.copy()
    test_df['heart_disease'] = y_test.values

    # Simpan
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    print(f"\n[INFO] Files saved to '{output_dir}/':")
    print(f"  - train.csv ({len(train_df)} rows)")
    print(f"  - test.csv ({len(test_df)} rows)")
    print(f"  - scaler.pkl")

    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler
    }


def preprocess_pipeline(filepath: str = None,
                        output_dir: str = 'heart_disease_preprocessing'):
    """Jalankan seluruh pipeline preprocessing."""
    print("=" * 60)
    print("HEART DISEASE PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    df = load_data(filepath)

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Remove Outliers
    df = remove_outliers(df)

    # Step 4: Create Binary Target
    df = create_binary_target(df)

    # Step 5: Scale & Split
    result = scale_and_split(df, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)

    return result


if __name__ == '__main__':
    output = os.path.join('preprocessing', 'heart_disease_preprocessing') \
        if os.path.exists('preprocessing') \
        else 'heart_disease_preprocessing'
    preprocess_pipeline(output_dir=output)
