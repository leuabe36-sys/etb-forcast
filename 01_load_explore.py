import pandas as pd
xls = pd.ExcelFile('Historical_ETB_Conversion_Rates_10Y.xlsx')
data = {}
for sheet in ['USD-ETB','EUR-ETB','GBP-ETB','JPY-ETB']:
    df = pd.read_excel(xls, sheet)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    data[sheet] = df
    print(sheet, df.shape, df['Date'].min(), df['Date'].max())
    # check frequency
    diffs = df['Date'].diff().dt.days.dropna()
    print('  median gap (days):', diffs.median(), 'max gap:', diffs.max())

import pickle
with open('data.pkl','wb') as f:
    pickle.dump(data, f)
