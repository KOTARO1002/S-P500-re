import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # CSVファイルを読み込む関数
    return pd.read_csv(filepath)

def feature_selection(df, target):
    # 数値データのみを選択し、目的変数を除外
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop(columns=[target], errors='ignore')  # 目的変数を除外

    # 相関が高い特徴を削除
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(to_drop, axis=1, inplace=True)

    return df

def scale_features(X):
    # 特徴のスケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_data(X, y, test_size=0.2, random_state=42):
    # データを訓練セットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def main():
    # データの読み込み
    df = load_data('sp500_unemployment_data_cleaned.csv')
    
    # 列名を確認
    print("データフレームの列名:")
    print(df.columns)

    # 目的変数（予測する変数）を指定
    target = 'Close'
    if target not in df.columns:
        print(f"エラー: 指定された目的変数 '{target}' がデータフレームに存在しません。")
        return

    # 特徴選択を実行（目的変数を渡す）
    df = feature_selection(df, target)
    
    # 特徴選択後、列名を再確認
    print("特徴選択後の列名:")
    print(df.columns)
    
    # 目的変数と説明変数を分割
    X = df.drop([target], axis=1)  # 目的変数の列を除外
    X = X.select_dtypes(include=[np.number])  # 数値型のデータのみを選択
    y = df[target]
    
    # 特徴のスケーリング
    X_scaled = scale_features(X)
    
    # データを訓練セットとテストセットに分割
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    print("訓練データのサンプル数:", X_train.shape[0])
    print("テストデータのサンプル数:", X_test.shape[0])

if __name__ == "__main__":
    main()
