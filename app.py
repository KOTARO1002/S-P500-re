from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import joblib
import os
import pandas as pd  # pandasをインポート

app = Flask(__name__)

# 保存されたモデルとスケーラーをロード
model = load_model('neural_network_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        date_str = request.form['date']
        unrate = float(request.form['unrate'])

        # 日付を分解して年と月を取得
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month

        # 特徴量を準備
        features = np.array([[year, month, unrate]])

        # デバッグログを追加
        app.logger.info(f"Received date: {date_str}, Unemployment rate: {unrate}")
        app.logger.info(f"Features before scaling: {features}")

        # フィーチャ名を設定（フィット時と一致させる）
        feature_names = ['Year', 'Month', 'UNRATE']
        features_df = pd.DataFrame(features, columns=feature_names)
        features_scaled = scaler.transform(features_df)

        # デバッグログを追加
        app.logger.info(f"Features after scaling: {features_scaled}")

        # 予測を実行
        prediction = model.predict(features_scaled)
        predicted_price = prediction[0][0]

        app.logger.info(f"Predicted price: {predicted_price}")

        return render_template('index.html', predicted_price=predicted_price)
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)