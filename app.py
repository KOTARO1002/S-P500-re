from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import joblib
import os
import time

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
        total_start_time = time.time()  # 全体の開始時間

        date_str = request.form['date']
        unrate = float(request.form['unrate'])

        # 日付を分解して年と月を取得
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month

        # 特徴量を準備
        features = np.array([[year, month, unrate]])
        app.logger.info(f"Received date: {date_str}, Unemployment rate: {unrate}")
        app.logger.info(f"Features before scaling: {features}")

        scaling_start_time = time.time()  # スケーリングの開始時間
        features_scaled = scaler.transform(features)
        scaling_end_time = time.time()  # スケーリングの終了時間
        app.logger.info(f"Features after scaling: {features_scaled}")

        prediction_start_time = time.time()  # 予測の開始時間
        prediction = model.predict(features_scaled)
        predicted_price = prediction[0][0]
        prediction_end_time = time.time()  # 予測の終了時間

        total_end_time = time.time()  # 全体の終了時間

        # 各ステップの時間をログに出力
        app.logger.info(f"Scaling Time: {scaling_end_time - scaling_start_time} seconds")
        app.logger.info(f"Prediction Time: {prediction_end_time - prediction_start_time} seconds")
        app.logger.info(f"Total Time: {total_end_time - total_start_time} seconds")

        return render_template('index.html', predicted_price=predicted_price)
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)