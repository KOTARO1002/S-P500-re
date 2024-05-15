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
        start_time = time.time()
        
        date_str = request.form['date']
        unrate = float(request.form['unrate'])

        # 日付を分解して年と月を取得
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month

        # 特徴量を準備
        features = np.array([[year, month, unrate]])
        print(f"Received date: {date_str}, Unemployment rate: {unrate}")
        print(f"Features before scaling: {features}")

        # 特徴量のスケーリング
        features_scaled_start = time.time()
        features_scaled = scaler.transform(features)
        features_scaled_end = time.time()
        print(f"Features after scaling: {features_scaled}")
        print(f"Scaling time: {features_scaled_end - features_scaled_start} seconds")

        # 予測を実行
        prediction_start = time.time()
        prediction = model.predict(features_scaled)
        prediction_end = time.time()
        predicted_price = prediction[0][0]
        print(f"Prediction: {predicted_price}")
        print(f"Prediction time: {prediction_end - prediction_start} seconds")

        end_time = time.time()
        print(f"Total request handling time: {end_time - start_time} seconds")

        return render_template('index.html', predicted_price=predicted_price)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)