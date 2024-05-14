from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import joblib

app = Flask(__name__)

# 保存されたモデルとスケーラーをロード
model = load_model('neural_network_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    unrate = float(request.form['unrate'])

    # 日付を分解して年と月を取得
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    month = date.month

    # 特徴量を準備
    features = np.array([[year, month, unrate]])
    features_scaled = scaler.transform(features)

    # 予測を実行
    prediction = model.predict(features_scaled)
    predicted_price = prediction[0][0]

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
