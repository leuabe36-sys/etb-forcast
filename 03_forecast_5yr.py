import pandas as pd, numpy as np, pickle
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

with open('data.pkl','rb') as f:
    data = pickle.load(f)

results = {}
forecast_horizon_years = 5
REGIME_START = pd.Timestamp('2024-07-29')  # ETB floated late July 2024

for sheet, df in data.items():
    col = df.columns[1]
    s = df.set_index('Date')[col]
    weekly_full = s.resample('W').last().ffill()

    # post-float regime only, for the model
    weekly = weekly_full[weekly_full.index >= REGIME_START]
    log_weekly = np.log(weekly)

    periods = forecast_horizon_years * 52

    model = ETSModel(log_weekly, error='add', trend='add', damped_trend=True, seasonal=None)
    fit = model.fit(disp=False)

    sim = fit.simulate(nsimulations=periods, repetitions=1000, anchor='end')
    fc_mean = np.exp(sim.mean(axis=1))
    fc_lower = np.exp(sim.quantile(0.1, axis=1))
    fc_upper = np.exp(sim.quantile(0.9, axis=1))

    last_date = weekly_full.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=periods, freq='W')

    results[sheet] = {
        'history_full': weekly_full,
        'history_regime': weekly,
        'future_dates': future_dates,
        'mean': fc_mean.values,
        'lower': fc_lower.values,
        'upper': fc_upper.values,
        'last_actual': weekly_full.iloc[-1],
    }
    print(f"{sheet}: last={weekly_full.iloc[-1]:.3f} | "
          f"1yr={fc_mean.values[51]:.2f} (CI {fc_lower.values[51]:.1f}-{fc_upper.values[51]:.1f}) | "
          f"3yr={fc_mean.values[155]:.2f} (CI {fc_lower.values[155]:.1f}-{fc_upper.values[155]:.1f}) | "
          f"5yr={fc_mean.values[-1]:.2f} (CI {fc_lower.values[-1]:.1f}-{fc_upper.values[-1]:.1f})")

with open('results2.pkl','wb') as f:
    pickle.dump(results, f)
