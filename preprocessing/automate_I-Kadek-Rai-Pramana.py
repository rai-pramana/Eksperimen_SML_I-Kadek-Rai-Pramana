"""
Automated Preprocessing Script for Wine Quality Dataset
========================================================
Script ini mengkonversi langkah-langkah preprocessing dari notebook eksperimen
menjadi pipeline otomatis yang menghasilkan data siap latih.

Author: I Kadek Rai Pramana
Dataset: Wine Quality (Red) - UCI ML Repository
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse
import joblib


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw wine quality dataset dari file CSV.

    Args:
        filepath: Path ke file CSV raw dataset.

    Returns:
        DataFrame berisi data mentah wine quality.
    """
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath, sep=';')
    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data: handle missing values dan duplicates.

    Args:
        df: DataFrame mentah.

    Returns:
        DataFrame yang sudah dibersihkan.
    """
    initial_shape = df.shape[0]

    # Cek dan hapus missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[INFO] Found {missing_count} missing values, dropping rows...")
        df = df.dropna()

    # Hapus duplikat
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"[INFO] Found {duplicate_count} duplicate rows, removing...")
        df = df.drop_duplicates()

    final_shape = df.shape[0]
    print(f"[INFO] Cleaning: {initial_shape} -> {final_shape} rows ({initial_shape - final_shape} removed)")

    return df.reset_index(drop=True)


def handle_outliers(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Handle outliers menggunakan metode IQR (Interquartile Range).

    Args:
        df: DataFrame input.
        columns: List kolom yang akan di-handle outliersnya.

    Returns:
        DataFrame tanpa outliers.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target column
        columns = [c for c in columns if c != 'quality']

    initial_shape = df.shape[0]

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    final_shape = df.shape[0]
    print(f"[INFO] Outlier removal: {initial_shape} -> {final_shape} rows ({initial_shape - final_shape} removed)")

    return df.reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering: transformasi target variable menjadi binary classification.
    - quality >= 7 -> 1 (good/high quality)
    - quality < 7  -> 0 (not good/standard quality)

    Args:
        df: DataFrame input.

    Returns:
        DataFrame dengan kolom target baru 'quality_label'.
    """
    df = df.copy()
    df['quality_label'] = (df['quality'] >= 7).astype(int)

    class_dist = df['quality_label'].value_counts()
    print(f"[INFO] Target distribution:")
    print(f"       - Not Good (0): {class_dist.get(0, 0)}")
    print(f"       - Good (1): {class_dist.get(1, 0)}")

    return df


def preprocess_pipeline(df: pd.DataFrame, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline preprocessing lengkap:
    1. Cleaning data
    2. Handle outliers
    3. Feature engineering
    4. Train-test split
    5. Feature scaling
    6. Simpan hasil

    Args:
        df: DataFrame mentah.
        output_dir: Direktori output untuk menyimpan data preprocessing.
        test_size: Proporsi data test.
        random_state: Random seed untuk reproducibility.

    Returns:
        Tuple (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n" + "=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Clean data
    print("\n[STEP 1] Cleaning data...")
    df = clean_data(df)

    # Step 2: Handle outliers
    print("\n[STEP 2] Handling outliers...")
    df = handle_outliers(df)

    # Step 3: Feature engineering
    print("\n[STEP 3] Feature engineering...")
    df = feature_engineering(df)

    # Step 4: Separate features and target
    print("\n[STEP 4] Splitting features and target...")
    feature_cols = [col for col in df.columns if col not in ['quality', 'quality_label']]
    X = df[feature_cols]
    y = df['quality_label']

    # Step 5: Train-test split (stratified)
    print("\n[STEP 5] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} samples")
    print(f"       Test:  {X_test.shape[0]} samples")

    # Step 6: Feature scaling
    print("\n[STEP 6] Feature scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    # Step 7: Save preprocessed data
    print(f"\n[STEP 7] Saving preprocessed data to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train_scaled.reset_index(drop=True),
                          y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled.reset_index(drop=True),
                         y_test.reset_index(drop=True)], axis=1)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"       Train saved: {train_df.shape} -> {train_path}")
    print(f"       Test saved:  {test_df.shape} -> {test_path}")
    print(f"       Scaler saved: {scaler_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    """Main entry point untuk menjalankan preprocessing secara otomatis."""
    parser = argparse.ArgumentParser(
        description='Automated preprocessing for Wine Quality dataset'
    )
    parser.add_argument(
        '--input', type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'wine_quality_raw.csv'),
        help='Path ke file raw dataset (default: ../wine_quality_raw.csv)'
    )
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(os.path.dirname(__file__), 'wine_quality_preprocessing'),
        help='Direktori output untuk data preprocessing'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Proporsi data test (default: 0.2)'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()

    # Load data
    df = load_data(args.input)

    # Run preprocessing pipeline
    preprocess_pipeline(df, args.output, args.test_size, args.random_state)


if __name__ == '__main__':
    main()
