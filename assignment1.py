import pandas as pd
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

pred = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Date'] = pd.to_datetime(data['Timestamp'])

px.line(data, x="Date", y='trips')

taxi_trips = data['trips']
taxi_trips.index = data['Date']
taxi_trips.index.freq = taxi_trips.index.inferred_freq

# Add trend component
trend = ExponentialSmoothing(taxi_trips, trend = 'add', seasonal = 'add').fit()

#Linear trend with damping
damptrend = ExponentialSmoothing(taxi_trips, trend = 'mul', seasonal = 'add', damped = True).fit(use_brute=True)

prediction = len(pred)
trend_fc = trend.forecast(prediction)
dtrend_fc = damptrend.forecast(prediction)

import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([taxi_trips.values, trend.fittedvalues.values, damptrend.fittedvalues.values]).T
smoothData.columns = ['Truth', 'Trend', 'Damped Trend']
smoothData.index = taxi_trips.index

fig = px.line(smoothData, y = ['Truth', 'Trend', 'Damped Trend'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Trend': 'red',
                            'Damped Trend': 'green'
                           },
              title='Linear and Damped Trends'
       )

fig.update_xaxes(range=[smoothData.index[-100], trend_fc.index[-1]])
fig.update_yaxes(range=[0, 25000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=trend_fc.index, y = trend_fc.values, name='Forecast Trend', line={'color':'red'}))
fig.add_trace(go.Scatter(x=dtrend_fc.index, y = dtrend_fc.values, name='Forecast Damped Trend', line={'color':'green'}))