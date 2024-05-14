import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

def get_data_yfinance(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        df.reset_index(inplace=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        print(f"{symbol} のデータ取得に成功しました。")
        return df
    except Exception as e:
        print(f"{symbol} のデータ取得中にエラーが発生しました: {e}")
        return None

def get_data_fred(symbol, start, end):
    try:
        df = web.DataReader(symbol, 'fred', start, end)
        df.reset_index(inplace=True)
        df['Year'] = df['DATE'].dt.year
        df['Month'] = df['DATE'].dt.month
        print(f"{symbol} のデータ取得に成功しました。")
        return df
    except Exception as e:
        print(f"{symbol} のデータ取得中にエラーが発生しました: {e}")
        return None

def main():
    start = '2010-01-01'
    end = '2024-01-01'

    sp500 = get_data_yfinance('^GSPC', start, end)
    unemployment = get_data_fred('UNRATE', start, end)

    if sp500 is not None and unemployment is not None:
        # 月次データに集約
        sp500_monthly = sp500.groupby(['Year', 'Month']).agg({'Close': 'mean'}).reset_index()
        # マージするキーを年と月に限定
        combined_data = pd.merge(sp500_monthly, unemployment, on=['Year', 'Month'], how='inner')
        print("データの統合に成功しました。")
        print(combined_data.head())
        combined_data.to_csv('sp500_unemployment_data.csv', index=False)
    else:
        print("データの統合に失敗しました。")

if __name__ == "__main__":
    main()
