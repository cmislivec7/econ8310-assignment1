import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

pred = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Date'] = pd.to_datetime(data['Timestamp'])

px.line(data, x="Date", y='trips')

taxi_trips = data['trips']
taxi_trips.index = data['Date']
taxi_trips.index.freq = taxi_trips.index.inferred_freq

# Add model
model =  ExponentialSmoothing(taxi_trips, trend = 'add', seasonal = 'add', damped = True).fit(use_brute=True)

prediction = len(pred)
modelFit = model.forecast(prediction)

import plotly.graph_objects as go

# Plotting our data

smoothData = pd.DataFrame([taxi_trips.values, model.fittedvalues.values]).T
smoothData.columns = ['Truth', 'Model']
smoothData.index = taxi_trips.index

fig = px.line(smoothData, y = ['Truth', 'Model'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Model': 'red'
                           },
              title='Linear and Damped Trends'
       )

fig.update_xaxes(range=[smoothData.index[-100], modelFit.index[-1]])
fig.update_yaxes(range=[0, 25000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=modelFit.index, y = modelFit.values, name='Forecast Trend', line={'color':'red'}))
