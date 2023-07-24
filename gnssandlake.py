import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd

# Enable interactive mode
plt.ion()

years1 = [1918, 1919, 1920, 
         1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930,
         1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940,
         1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950,
         1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960,
         1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970,
         1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980,
         1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990,
         1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
         2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
         2021, 2022, 2023]

# Yearly average water levels data michiga huron
data1 = [2.836851466668, 2.37, 1.978365, 1.53, 1.39, 0.79, 0.54, -0.34, -0.45, 0.414497933332, 1.382345733332, 2.87, 2.06, 0.32, -0.27,
        -0.51, -0.837689333332, -0.43, -0.27, -0.3264251, 0.39, 0.810932766668, 0.39, 0.33, 1.02, 1.98, 1.89, 1.80, 1.901812066668, 1.79,
        1.67, 0.619550433332, 0.78, 2.34, 3.49, 2.989957333332, 2.648203166668, 2.30, 1.3714096, 0.791794533332, 0.15, -0.05, 1.4944411,
        1.16, 0.6578269, -0.33, -1.11, -0.348297366668, 0.455508433332, 0.91, 1.393281866668, 2.210757833332, 2.15, 2.57, 2.84,
        3.613316933332, 3.514891733332, 3.12, 2.88, 1.58, 1.87, 2.53, 2.56, 2.218959933332, 1.89, 2.661873333332, 2.8641918, 3.62, 4.17,
        3.1102548, 1.778780566668, 1.24, 1.08, 1.467100766668, 1.50, 2.210757833332, 2.153343133332, 1.66, 2.07, 3.16, 2.28, 0.70,
        -0.143244866668, -0.23, 0.32, -0.43, 0.291466433332, 0.2231156, -0.02, -0.258074266668, -0.0557558,
        0.775390333332, 0.29, 0.048137466668, -0.345563333332, -0.42, 0.92, 1.874471733332, 2.22, 2.4540868, 2.77, 3.68, 4.212070233332,
        2.99, 2.06, 1.58]
# Multiply all the values in data1 by 0.304
data1 = [value * 0.304 for value in data1]

data2 = pd.read_csv('/Users/ash/Desktop/NOAA/project1/Calumet.csv')
# Convert the date column to datetime format
data2['Date'] = pd.to_datetime(data2['Date'])
data2['MSL (m)'] = (data2['MSL (m)'] - 176.0)



# Calculate yearly averages for data1
data1_yearly_avg = pd.DataFrame({'Date': years1, 'Vertical': data1})
data1_yearly_avg = data1_yearly_avg.groupby('Date')['Vertical'].mean()

# Calculate yearly averages for data2
data2_yearly_avg = data2.groupby(data2['Date'].dt.year)['MSL (m)'].mean()



# Perform linear regression for data1
regression_model1 = LinearRegression()
X1 = data1_yearly_avg.index.values.reshape(-1, 1)
y1 = data1_yearly_avg.values.reshape(-1, 1)
regression_model1.fit(X1, y1)
y_pred1 = regression_model1.predict(X1)

# Perform linear regression for data2
regression_model2 = LinearRegression()
X2 = data2_yearly_avg.index.values.reshape(-1, 1)
y2 = data2_yearly_avg.values.reshape(-1, 1)
regression_model2.fit(X2, y2)
y_pred2 = regression_model2.predict(X2)



# Add the slopes to the plot's legeng
slope1 = regression_model1.coef_[0][0]
slope2 = regression_model2.coef_[0][0]


# Configure the axes
fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(1918, 2022)  # Set the x-axis limits
ax.set_ylim(0.86, 0.95)  # Set the y-axis limits
ax.set_facecolor('#f8f8f8')  # Give plot a gray background like ggplot
[ax.spines[side].set_visible(False) for side in ax.spines]  # Remove border around plot

# Plot the yearly average and best fit lines for data1
ax.plot(data1_yearly_avg.index, data1_yearly_avg.values, color='dodgerblue', label='Lake Michigan Data 1')
ax.plot(data1_yearly_avg.index, y_pred1.flatten(), color='blue', linestyle='--', label='Line Of Best Fit Data 1')

# Plot the yearly average and best fit lines for data2
ax.plot(data2_yearly_avg.index[:-1], data2_yearly_avg.values[:-1], color='orangered', label='Lake Michigan Calumet')
ax.plot(data2_yearly_avg.index[:-1], y_pred2[:-1], color='darkorange', linestyle='--', label='Line Of Best Fit Calumet')



ax.legend(title='Slopes', labels=[f'Calumet: {slope2:.3f}'])


# Convert the Matplotlib figure to Plotly format
fig_plotly = go.Figure()
fig_plotly.add_trace(go.Scatter(x=data1_yearly_avg.index, y=data1_yearly_avg.values, name='Michigan-Huron Water Levels', line=dict(color='dodgerblue')))
fig_plotly.add_trace(go.Scatter(x=data2_yearly_avg.index[:-1], y=data2_yearly_avg.values[:-1], name='Calumet Water Levels', line=dict(color='orangered')))
fig_plotly.add_trace(go.Scatter(x=data1_yearly_avg.index[:-1], y=y_pred1.flatten(), name='Line Of Best Fit Michigan-Huron Water Levels', line=dict(color='blue', dash='dash')))
fig_plotly.add_trace(go.Scatter(x=data2_yearly_avg.index[:-1], y=y_pred2.flatten(), name='Line Of Best Fit Calumet Water Levels', line=dict(color='darkorange', dash='dash')))

fig_plotly.update_layout(
    title={'text': 'Lake Michigan-Huron Water Levels vs Calumet Water Levels', 'font': dict(color='RebeccaPurple')},
    xaxis_title={'text': 'Year', 'font': dict(color='RebeccaPurple')},
    yaxis_title={'text': ' Water Level Minus Datums (m)', 'font': dict(color='RebeccaPurple')},
    hovermode='closest',
    font=dict(family='Courier New, monospace', size=16),
    xaxis_range=[1918, 2022]
)

# Add the slope text annotations to the side of the graph
slope_annotation1 = f'Lake Michigan-Huron Water Levels Slope: {slope1:.3f}'
slope_annotation2 = f'Calumet Water Levels Slope: {slope2:.3f}'
fig_plotly.add_annotation(
    x=0.05,
    y=0.92,
    text=slope_annotation1,
    showarrow=False,
    font=dict(size=15),
    align='left',
    xref='paper',
    yref='paper'
)
fig_plotly.add_annotation(
    x=0.05,
    y=0.88,
    text=slope_annotation2,
    showarrow=False,
    font=dict(size=15),
    align='left',
    xref='paper',
    yref='paper'
)


# Show the graph
fig_plotly.show()

# Save the figure as an HTML file
fig_plotly.write_html('/Users/ash/Desktop/NOAA/project1/gnssandlake.html')


