!pip install wheel
!pip install pandas numpy
from IPython import get_ipython
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# %%
data_rows = []
content = """date location new_cases new_deaths total_cases total_deaths
01/22/2020 World 0 0 557 17
01/23/2020 World 95 0 652 18
01/24/2020 World 275 0 927 26
01/25/2020 World 496 0 1423 42
01/26/2020 World 686 0 2109 56
01/27/2020 World 809 0 2918 82
01/28/2020 World 1348 0 4266 103
01/29/2020 World 1761 0 6027 132
01/30/2020 World 1970 0 7997 171
01/31/2020 World 2103 0 10100 213
02/01/2020 World 2590 0 12690 259
02/02/2020 World 2777 0 15467 305
02/03/2020 World 2829 0 18296 362
02/04/2020 World 3242 0 21538 426
02/05/2020 World 3881 0 25419 492
02/06/2020 World 3174 0 28593 566
02/07/2020 World 3138 0 31731 636
02/08/2020 World 2665 0 34396 717
02/09/2020 World 2670 0 37066 806
02/10/2020 World 2484 0 39550 910
02/11/2020 World 2020 0 41570 1013
02/12/2020 World 2496 0 44066 1118
02/13/2020 World 14994 242 59060 1360
02/14/2020 World 5223 122 64283 1482
02/15/2020 World 2103 146 66386 1628
02/16/2020 World 2065 110 68451 1738
02/17/2020 World 1886 98 70337 1836
02/18/2020 World 1825 100 72162 1936
02/19/2020 World 691 0 72853 2007
02/20/2020 World 444 0 73297 2012
02/21/2020 World 555 5 73852 2017
02/22/2020 World 425 7 74277 2024
02/23/2020 World 688 14 74965 2038
02/24/2020 World 516 15 75481 2053
02/25/2020 World 887 23 76368 2076
02/26/2020 World 1360 29 77728 2105
02/27/2020 World 1200 16 78928 2121
02/28/2020 World 1325 46 80253 2167
02/29/2020 World 1757 40 82010 2207
03/01/2020 World 1719 41 83729 2248
03/02/2020 World 2369 58 86098 2306
03/03/2020 World 2300 65 88398 2371
03/04/2020 World 2264 78 90662 2449
03/05/2020 World 2246 88 92908 2537
03/06/2020 World 3609 114 96517 2651
03/07/2020 World 3715 106 100232 2757
03/08/2020 World 3695 108 103927 2865
03/09/2020 World 4285 185 108212 3050
03/10/2020 World 4233 184 112445 3234
03/11/2020 World 4613 236 117058 3470
03/12/2020 World 7575 334 124633 3804
03/13/2020 World 8350 424 132983 4228
03/14/2020 World 11622 554 144605 4782
03/15/2020 World 11774 561 156379 5343
03/16/2020 World 13892 787 170271 6130
03/17/2020 World 17280 934 187551 7064
01/22/2020 Italy 0 0 0 0
01/23/2020 Italy 0 0 0 0
01/24/2020 Italy 0 0 0 0
01/25/2020 Italy 0 0 0 0
01/26/2020 Italy 0 0 0 0
01/27/2020 Italy 0 0 0 0
01/28/2020 Italy 0 0 0 0
01/29/2020 Italy 0 0 0 0
01/30/2020 Italy 2 0 2 0
01/31/2020 Italy 0 0 2 0
02/01/2020 Italy 0 0 2 0
02/02/2020 Italy 0 0 2 0
02/03/2020 Italy 0 0 2 0
02/04/2020 Italy 0 0 2 0
02/05/2020 Italy 0 0 2 0
02/06/2020 Italy 0 0 2 0
02/07/2020 Italy 0 0 2 0
02/08/2020 Italy 0 0 2 0
02/09/2020 Italy 0 0 2 0
02/10/2020 Italy 0 0 2 0
02/11/2020 Italy 0 0 2 0
02/12/2020 Italy 0 0 2 0
02/13/2020 Italy 0 0 2 0
02/14/2020 Italy 0 0 2 0
02/15/2020 Italy 0 0 2 0
02/16/2020 Italy 0 0 2 0
02/17/2020 Italy 0 0 2 0
02/18/2020 Italy 0 0 2 0
02/19/2020 Italy 0 0 2 0
02/20/2020 Italy 0 0 3 0
02/21/2020 Italy 19 1 22 1
02/22/2020 Italy 24 1 46 2
02/23/2020 Italy 80 2 126 4
02/24/2020 Italy 238 5 366 9
02/25/2020 Italy 93 4 459 13
02/26/2020 Italy 78 0 537 17
02/27/2020 Italy 250 5 787 21
02/28/2020 Italy 238 8 1025 29
02/29/2020 Italy 240 4 1265 33
03/01/2020 Italy 566 6 1831 39
03/02/2020 Italy 561 6 2392 52
03/03/2020 Italy 347 11 2735 79
03/04/2020 Italy 587 28 3341 134
03/05/2020 Italy 769 41 4161 148
03/06/2020 Italy 620 49 4781 237
03/07/2020 Italy 1247 36 6013 233
03/08/2020 Italy 1492 133 7428 366
03/09/2020 Italy 1797 97 9200 463
03/10/2020 Italy 977 168 10149 631
03/11/2020 Italy 2313 196 12462 827
03/12/2020 Italy 2651 189 15113 1016
03/13/2020 Italy 2116 250 17237 1266
03/14/2020 Italy 2651 174 19898 1440
03/15/2020 Italy 2800 179 22513 1624
03/16/2020 Italy 3233 347 25595 1809
03/17/2020 Italy 3526 345 28710 2158
"""  # Replace with actual content

for line in content.split('\n'):
    if line.startswith('=====') or not line.strip() or line.startswith('date location'):
        continue
    line_clean = line.replace('|', ' ').strip()
    tokens = line_clean.split()
    if len(tokens) < 5 or '/' not in tokens[0]:
        continue

    date_str = tokens[0]
    location = ' '.join(tokens[1:-4])
    num_values = tokens[-4:]

    try:
        new_cases = pd.to_numeric(num_values[0], errors='coerce')
        new_deaths = pd.to_numeric(num_values[1], errors='coerce')
        total_cases = pd.to_numeric(num_values[2], errors='coerce')
        total_deaths = pd.to_numeric(num_values[3], errors='coerce')
        data_rows.append([
            date_str, location, new_cases, new_deaths, total_cases, total_deaths
        ])
    except:
        continue

df = pd.DataFrame(data_rows, columns=[
    'date', 'location', 'new_cases', 'new_deaths', 'total_cases', 'total_deaths'
])
df['date'] = pd.to_datetime(df['date'])

# Summary Statistics (as of last date)
summary = df.sort_values('date').groupby('location').last()

# Check if 'World' is in the index before accessing it
if 'World' in summary.index:
    global_summary = summary.loc['World'][['total_cases', 'total_deaths']]
    print("Global Summary (as of 2020-03-17):")
    print(f"Total Cases: {int(global_summary['total_cases']):,}")
    print(f"Total Deaths: {int(global_summary['total_deaths']):,}\n")
else:
    print("Warning: 'World' data not found in the dataset.")
    global_summary = None # Set to None or handle appropriately

top_countries = summary.drop(index=['World', 'International'], errors='ignore').nlargest(5, 'total_cases')

print("Top 5 Countries by Total Cases:")
# Applymap might be deprecated in newer pandas versions, use apply with axis=1 instead
# print(top_countries[['total_cases', 'total_deaths']].applymap(lambda x: f"{int(x):,}"))
print(top_countries[['total_cases', 'total_deaths']].apply(lambda x: x.apply(lambda val: f"{int(val):,}"), axis=1))


# Forecasting function
def forecast_covid_data(location_name, days=7):
    loc_df = df[df['location'] == location_name].sort_values('date')
    series = loc_df.set_index('date')['new_cases'].dropna()

    # Handle zeros in data
    if (series == 0).any():
        series = series.replace(0, np.nan).interpolate()

    # Check if there are enough data points to fit the ARIMA model
    if len(series) < 2:
        print(f"Insufficient data to forecast for {location_name}.")
        return pd.DataFrame(columns=['date', 'forecasted_cases'])

    # Fit ARIMA model (parameters optimized for COVID data patterns)
    try:
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)

        # Generate forecast dates
        last_date = series.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series, label='Historical Data', marker='o')
        plt.plot(forecast_dates, forecast, label='Forecast', marker='x', color='red')
        plt.title(f'Daily New Cases Forecast for {location_name}')
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_cases': forecast.round().astype(int)
        })
        return forecast_df
    except Exception as e:
        print(f"Error forecasting data for {location_name}: {e}")
        return pd.DataFrame(columns=['date', 'forecasted_cases'])
        # Get a list of all unique locations (countries)
all_locations = df['location'].unique()

# Get a list of all unique locations (countries)
all_locations = df['location'].unique()

# Iterate through each location and generate the forecast
print("\nARIMA Forecast for All Countries (Next 7 Days):")
for location in all_locations:
    print(f"\n--- {location} ---")
    location_forecast = forecast_covid_data(location)
    if not location_forecast.empty:
        display(location_forecast[['date', 'forecasted_cases']].to_string(index=False))
    else:
        print("Could not generate forecast for this location.")