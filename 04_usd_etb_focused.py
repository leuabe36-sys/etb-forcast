"""
USD/ETB focused analysis
Combines: load, deep backtest (multiple holdout windows), and 5yr forecast
into one script, for a single pair.
"""
import pandas as pd, numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# --- Load ---
xls = pd.ExcelFile('Historical_ETB_Conversion_Rates_10Y.xlsx')
df = pd.read_excel(xls, 'USD-ETB')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
s = df.set_index('Date')['USD to ETB']
weekly_full = s.resample('W').last().ffill()

REGIME_START = pd.Timestamp('2024-07-29')
weekly = weekly_full[weekly_full.index >= REGIME_START]

print(f"Post-float weekly points: {len(weekly)}  ({weekly.index[0].date()} to {weekly.index[-1].date()})")
print()

# --- Backtest at multiple horizons ---
print("Backtest — model vs naive, at different holdout lengths:")
print(f"{'Holdout':<10}{'Naive MAPE':>12}{'Model MAPE':>13}{'Model wins?':>13}")
for holdout in [8, 13, 26, 52]:
    if holdout >= len(weekly) - 20:
        continue
    train = weekly.iloc[:-holdout]
    test = weekly.iloc[-holdout:]
    log_train = np.log(train)
    fit = ETSModel(log_train, error='add', trend='add', damped_trend=True, seasonal=None).fit(disp=False)
    fc = np.exp(fit.forecast(holdout))
    naive = pd.Series(train.iloc[-1], index=test.index)
    model_mape = (np.abs((fc.values - test.values) / test.values)).mean() * 100
    naive_mape = (np.abs((naive.values - test.values) / test.values)).mean() * 100
    wins = "yes" if model_mape < naive_mape else "no"
    print(f"{holdout}wk{'':<6}{naive_mape:>11.2f}%{model_mape:>12.2f}%{wins:>13}")

print()

# --- Full 5yr forecast on all available post-float data ---
log_weekly = np.log(weekly)
fit = ETSModel(log_weekly, error='add', trend='add', damped_trend=True, seasonal=None).fit(disp=False)
periods = 5 * 52
sim = fit.simulate(nsimulations=periods, repetitions=1000, anchor='end')
fc_mean = np.exp(sim.mean(axis=1))
fc_lower = np.exp(sim.quantile(0.1, axis=1))
fc_upper = np.exp(sim.quantile(0.9, axis=1))

print("USD/ETB forecast:")
for label, wk in [('3 months', 13), ('6 months', 26), ('1 year', 51), ('2 years', 103), ('3 years', 155), ('5 years', 259)]:
    print(f"  {label:<10} {fc_mean.values[wk]:>8.2f}  (90% CI: {fc_lower.values[wk]:.1f} - {fc_upper.values[wk]:.1f})")
