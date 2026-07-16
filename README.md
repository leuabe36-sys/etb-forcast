# ETB forecasting — learn & test kit

Files:
- `01_load_explore.py`  — load the workbook, resample, plot the trend, spot the 2024 regime break
- `02_backtest.py`      — the core testing loop: hold out recent weeks, forecast, score against reality
- `03_forecast_5yr.py`  — fit on all available data, project forward with uncertainty bands

## The one idea that matters
Never trust a forecast's accuracy claim without a backtest:
1. Pick a cutoff date. Hide everything after it.
2. Fit the model only on data before the cutoff.
3. Forecast forward to the cutoff+N.
4. Compare to what *actually* happened (which you hid, but still have).
5. Compare that error to a dumb baseline (e.g. "tomorrow = today"). If you don't beat the dumb baseline, the model isn't earning its complexity.

## To experiment
- Change `HOLDOUT_WEEKS` in `02_backtest.py` — does the model still win at 12 weeks? 52 weeks?
- Swap `trend='add'` for `trend='mul'` or drop `damped_trend` — compare MAPE.
- Try `statsmodels.tsa.arima.model.ARIMA` instead of ETS and compare.
- The gap between "backtest looks great" and "5-year forecast is trustworthy" is real — a backtest only proves the model works over the horizon you tested. Extrapolating that confidence to 5 years is a judgment call, not a proven fact.
