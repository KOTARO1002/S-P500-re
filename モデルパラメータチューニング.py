import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

# データの読み込み
data = pd.read_csv('integrated_data_with_features.csv')

# NaN を含む行を削除
data.dropna(inplace=True)

# 日付データの処理
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

# 日付列の削除
data.drop('Date', axis=1, inplace=True)

# 目的変数と説明変数の指定
X = data[['year', 'month', 'day', 'Unemployment_Rate']]
y = data['S&P500_Close']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# パイプラインの作成
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=0))
])

# パラメータグリッドの再設定
param_grid = {
    'rf__n_estimators': [50, 100],  # 数を減らす
    'rf__max_features': ['sqrt', 'log2'],
    'rf__max_depth': [5, 10],  # 深さを制限する
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

# GridSearchCVの設定
grid_search = GridSearchCV(pipeline, param_grid, cv=10, verbose=2, n_jobs=-1)

# グリッドサーチの実行
grid_search.fit(X_train, y_train)

# 最適なパラメータとそのスコアを表示
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# テストデータでの評価
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# モデルの保存
dump(grid_search.best_estimator_, 'trained_model.joblib')
