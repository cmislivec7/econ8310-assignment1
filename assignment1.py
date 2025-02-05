import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES, SimpleExpSmoothing

test_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Date'] = pd.to_datetime(data['Timestamp'])
test_data['Date'] = pd.to_datetime(test_data['Timestamp'])


px.line(data, x="Date", y='trips')

taxi_trips = data['trips']
taxi_trips.index = data['Date']
taxi_trips.index.freq = taxi_trips.index.inferred_freq

model = ES(taxi_trips, trend = 'mul', seasonal = 'add', damped = True)#.fit()
modelFit = model.fit()
prediction = len(test_data)
forecast = modelFit.forecast(steps = 744)
pred = pd.DataFrame(forecast, columns= ["trips"])
import plotly.graph_objects as go

# Plotting our data"

smoothData = pd.DataFrame([taxi_trips.values, forecast.values]).T
smoothData.columns = ['Truth', 'Model']
smoothData.index = taxi_trips.index

fig = px.line(smoothData, y = ['Truth', 'Model'], 
        x = smoothData.index,
        color_discrete_map={"Truth": 'blue',
                           'Model': 'red'
                           },
              title='Linear and Damped Trends'
       )

fig.update_xaxes(range=[taxi_trips.index[-1000], forecast.index[-1]])
fig.update_yaxes(range=[0, 25000])


# Incorporating the Forecasts

fig.add_trace(go.Scatter(x=forecast.index, y = forecast.values, name='Forecast Trend', line={'color':'red'}))