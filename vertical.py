import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd

# Enable interactive mode
plt.ion()

years1 = [...]  # The list of years (data1) is provided in your original code.


data3 = pd.DataFrame({
    'Date': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Vertical': [0.9324085625, 0.9267443025, 0.9188609818, 0.9193130909, 0.9153166444, 0.9153006061, 0.9083119727, 0.9027900529, 0.9035817940, 0.9033872370, 0.9020906703, 0.8954964214, 0.8906804753, 0.8869811168, 0.8860495238, 0.8788295795, 0.8765370646, 0.8706148119, 0.8688887826, 0.8702171985]
})

# 

# Calculate yearly averages for data3
data3_yearly_avg = data3.groupby(data3['Date'])['Vertical'].mean()

# Perform linear regression for data3
regression_model3 = LinearRegression()
X3 = data3_yearly_avg.index.values.reshape(-1, 1)
y3 = data3_yearly_avg.values.reshape(-1, 1)
regression_model3.fit(X3, y3)
y_pred3 = regression_model3.predict(X3)

# Add the slope to the plot's legend
slope3 = regression_model3.coef_[0][0]


# Configure the axes
fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(1918, 2022)  # Set the x-axis limits
ax.set_ylim(0.86, 0.95)  # Set the y-axis limits
ax.set_facecolor('#f8f8f8')  # Give plot a gray background like ggplot
[ax.spines[side].set_visible(False) for side in ax.spines]  # Remove border around plot

# Plot the yearly average and best fit lines for data3
ax.plot(data3_yearly_avg.index, data3_yearly_avg.values, color='mediumorchid', label='GNSS')
ax.plot(data3_yearly_avg.index, y_pred3.flatten(), color='purple', linestyle='--', label='Line Of Best Fit GNSS')


# Convert the Matplotlib figure to Plotly format
fig_plotly = go.Figure()
fig_plotly.add_trace(go.Scatter(x=data3_yearly_avg.index, y=data3_yearly_avg.values, name='GNSS', line=dict(color='mediumorchid')))
fig_plotly.add_trace(go.Scatter(x=data3_yearly_avg.index, y=y_pred3.flatten(), name='Line Of Best Fit GNSS', line=dict(color='purple', dash='dash')))


fig_plotly.update_layout(
    title={'text': 'GNSS Vertical Displacement', 'font': dict(color='RebeccaPurple')},
    xaxis_title={'text': 'Year', 'font': dict(color='RebeccaPurple')},
    yaxis_title={'text': 'Vertical Displacement (m)', 'font': dict(color='RebeccaPurple')},
    hovermode='closest',
    font=dict(family='Courier New, monospace', size=16),
)

# ... (rest of the code remains unchanged)
slope_annotation3 = f'Vertical Displacement: {slope3:.3f}'
fig_plotly.add_annotation(
    x=0.05,
    y=0.92,
    text=slope_annotation3,
    showarrow=False,
    font=dict(size=18),
    align='left',
    xref='paper',
    yref='paper'
)

# ... (rest of the code remains unchanged)

# Show the graph
fig_plotly.show()

# Save the figure as an HTML file
fig_plotly.write_html('/Users/ash/Desktop/NOAA/project1/vertical.html')
