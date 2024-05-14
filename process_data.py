import pandas as pd

def main():
    # CSVファイルの読み込み
    df = pd.read_csv('sp500_unemployment_data.csv')
    print("初期データの先頭5行:")
    print(df.head())

    # 欠損値の確認
    print("\n欠損値の確認:")
    print(df.isnull().sum())

    # 数値列だけに欠損値処理を適用
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 欠損値の処理後の確認
    print("\n欠損値の処理後の確認:")
    print(df.isnull().sum())

    # 年月日の列を組み合わせて datetime 型の日付列を作成
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # 古い年、月、日の列を削除
    df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

    # データ型の確認
    print("\nデータ型の確認:")
    print(df.dtypes)

    # 変更を加えたデータフレームを新しいCSVファイルに保存
    df.to_csv('sp500_unemployment_data_cleaned.csv', index=False)

    print("\n処理後のデータの先頭5行:")
    print(df.head())

if __name__ == "__main__":
    main()