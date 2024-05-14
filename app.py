from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime
import joblib

app = Flask(__name__)
model = joblib.load('trained_model.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ユーザーからの入力を取得
        input_date = request.form['date']
        input_unemployment_rate = float(request.form['unemployment_rate'])

        # 日付を年、月、日に分解
        date_obj = datetime.strptime(input_date, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day

        # 予測用のデータフレームを作成
        input_data = {
            'year': [year],
            'month': [month],
            'day': [day],
            'Unemployment_Rate': [input_unemployment_rate]
        }
        input_df = pd.DataFrame(input_data)

        # 予測実行
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
