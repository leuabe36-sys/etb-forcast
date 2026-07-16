import pandas as pd, numpy as np, pickle
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

with open('data.pkl','rb') as f:
    data = pickle.load(f)

REGIME_START = pd.Timestamp('2024-07-29')
HOLDOUT_WEEKS = 26  # test on most recent 6 months

print(f"{'Pair':<10}{'Naive MAPE':>12}{'Model MAPE':>13}{'Naive MAE':>12}{'Model MAE':>12}")
for sheet, df in data.items():
    col = df.columns[1]
    s = df.set_index('Date')[col]
    weekly_full = s.resample('W').last().ffill()
    weekly = weekly_full[weekly_full.index >= REGIME_START]

    train = weekly.iloc[:-HOLDOUT_WEEKS]
    test = weekly.iloc[-HOLDOUT_WEEKS:]

    # Model forecast
    log_train = np.log(train)
    model = ETSModel(log_train, error='add', trend='add', damped_trend=True, seasonal=None)
    fit = model.fit(disp=False)
    fc = np.exp(fit.forecast(HOLDOUT_WEEKS))

    # Naive baseline: last value repeated
    naive = pd.Series(train.iloc[-1], index=test.index)

    model_mape = (np.abs((fc.values - test.values) / test.values)).mean() * 100
    naive_mape = (np.abs((naive.values - test.values) / test.values)).mean() * 100
    model_mae = np.abs(fc.values - test.values).mean()
    naive_mae = np.abs(naive.values - test.values).mean()

    print(f"{sheet:<10}{naive_mape:>11.2f}%{model_mape:>12.2f}%{naive_mae:>12.3f}{model_mae:>12.3f}")
