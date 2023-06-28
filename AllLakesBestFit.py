import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.ticker import AutoMinorLocator
from sklearn.linear_model import LinearRegression
import mpld3

    
# Define the data
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
# Yearly average water levels data Erie
data2= [1.7153726, 2.259445233332, 1.37, 1.75, 1.43, 0.87, 1.15, 0.33, 0.487791633332,
        1.07, 1.507586066668, 2.69, 2.55, 0.70, 0.799471433332, 0.61, -0.526534733332,
        -0.266801566668, -0.012536466668, 1.0756088, 1.14, 1.16, 0.9935878, 0.68,
        1.261523066668, 2.426221266668, 1.97, 2.28, 2.111807433332, 2.44, 2.390678833332,
        1.58, 2.1090734, 2.729698966668, 3.3721968, 2.75, 2.612135533332, 2.808985933332,
        2.0598608, 1.79, 1.037332333332, 1.17, 1.975105766668, 1.996978033332, 1.3626823,
        0.769397066668, 0.38, 0.7229185, 1.34, 1.860276366668, 2.335998166668, 2.95, 2.51,
        2.69, 3.30658, 4.10, 3.927205566668, 3.686610633332, 3.547174933332, 2.6422099, 2.89,
        2.95, 3.38, 2.97, 3.06, 3.37, 3.3803989, 4.06, 4.61, 3.8561207, 2.52, 2.41, 2.631273766668,
        2.74, 2.84, 3.32, 2.8718687, 2.59, 2.89, 4.01, 3.470622, 1.994244, 1.63, 1.38, 1.857542333332,
        1.55, 2.05, 2.2239028, 2.136413733332, 2.12, 2.19, 2.48, 1.85, 2.456295633332, 2.09, 1.8630104,
        2.357870433332, 2.71, 3.0113044, 3.552643, 3.83, 4.41, 4.5943097, 3.87, 3.25, 3.29]
# Yearly average water levels data Erie Ontario
data3 = [2.3446929, 2.62, 1.3440367, 1.666652633333, 1.64, 0.936665733333, 1.3112283, 0.7616876,
        0.8026981, 1.472536266667, 2.11, 3.04, 3.03, 0.66, 1.13, 0.387125033333, -0.45, -0.50, 0.02,
        1.108909833333, 1.1963989, 1.0979737, 0.88, 0.8191023, 1.177260666667, 3.20, 2.257203833333,
        2.96, 2.741127733333, 3.318008766667, 3.03, 1.91, 2.407575666667, 3.57, 3.963240633333,
        2.8122126, 2.70, 3.24, 2.51, 1.560025333333, 0.887453133333, 1.040559, 1.82, 1.49, 1.59,
        1.63, 0.42, 0.78, 1.75, 2.298214333333, 2.104097966667, 1.94, 1.73, 1.86, 2.55, 3.45,
        2.978988633333, 2.1396404, 2.91, 1.970130333333, 2.50, 2.24, 2.183384933333, 2.06, 1.9509921,
        2.18, 2.440384066667, 2.36, 3.2469239, 2.268139966667, 1.71, 1.978332433333, 2.12, 2.3364908,
        2.21, 2.90, 1.9591942, 1.907247566667, 2.35, 2.6399685, 2.47, 1.357706866667, 2.0412152, 1.78,
        2.07, 1.82, 2.27, 2.15, 2.21, 1.95, 2.352895, 2.35, 1.68, 2.11, 1.7623438, 1.90, 2.02, 1.9017795,
        2.11, 3.334412966667, 2.437650033333, 3.6324226, 2.85, 1.83, 2.15, 2.63]
# Yearly average water levels data Superior
data4 =[0.370929133332, 0.60, 0.70, 0.46, 0.33, 0.03, -0.28, -0.61, -0.89, 0.35, 0.841182866668, 0.93,
        0.734555566668, 0.36, 0.690811033332, 0.75, 0.93, 0.9669484, 0.80, 0.78, 1.13, 1.1145862, 0.55,
        0.74, 1.08, 1.29, 0.96, 1.07, 1.002490833332, 1.10, 0.584183733332, 0.75, 1.37, 1.59, 1.346979033332,
        0.871257233332, 0.641598433332, 0.420141733332, 0.36, 0.33, 0.29, 0.61, 0.658002633332, 0.32, 0.29, 0.27,
        0.414673666668, 0.40, 0.60, 0.56, 0.90, 0.97, 0.772832033332, 1.20, 1.11, 1.25, 1.3114366, 1.24, 0.882193366668,
        0.496694666668, 0.731821533332, 0.915001766668, 0.723619433332, 0.4748224, 0.67, 1.16, 1.0653736, 1.49, 1.6969353,
        0.830246733332, 0.3353867, 0.61, 0.14, 0.43, 0.5978539, 0.78, 0.7454917, 0.420141733332, 1.098182, 1.25, 0.357258966668,
        0.1631426, -0.12, -0.04, 0.23, -0.09, 0.27, 0.37, -0.10, -0.79, -0.01, 0.1549405, -0.23,
        -0.400068266668, -0.383664066668, 0.004568666668, 0.96, 1.17, 1.21, 1.36, 1.344245, 1.75, 1.57, 0.88, 0.78, 0.97]


# Create the figure and axis with a larger width
fig, ax = plt.subplots(figsize=(18, 8))

# Configure the axes
ax.set_xlim(1918, 2023)  # Set the x-axis limits
ax.set_ylim(-1, 5)  # Set the y-axis limits



# Initialize the line plots
line1, = ax.plot([], [], color='black',label='Average Water Level (Michigan-Huron)', alpha=.6, lw=3)
line2, = ax.plot([], [], color='red', label='Average Water Level (Erie)', alpha=.6, lw=3)
line3, = ax.plot([], [], color='green', label='Average Water Level (Ontario)', alpha=.6, lw=3)
line4, = ax.plot([], [], color='blue',label='Average Water Level (Superior)', alpha=.6, lw=3)



# Give plot a gray background like ggplot
ax.set_facecolor('#f8f8f8')

# Remove border around plot
[ax.spines[side].set_visible(False) for side in ax.spines]


# Enable interactive mode
plt.ion()

# Convert the Matplotlib figure to Plotly format
fig = go.Figure(data=[go.Scatter(x=years1, y=data1, name='Michigan Huron'),
                      go.Scatter(x=years1, y=data2, name='Erie'),
                      go.Scatter(x=years1, y=data3, name='Ontario'),
                      go.Scatter(x=years1, y=data4, name='Superior')])

fig.update_layout(
    title="Lake Level Changes",
    xaxis_title="Years",
    yaxis_title="Lake Level (ft)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()

#Perform linear regression (degree 1)
coefficients = np.polyfit(years1, data1, 1)
slope = coefficients[0]
intercept = coefficients[1]

coefficients2 = np.polyfit(years1, data2, 1)
slope = coefficients[0]
intercept = coefficients[1]

coefficients3 = np.polyfit(years1, data3, 1)
slope = coefficients[0]
intercept = coefficients[1]

coefficients4 = np.polyfit(years1, data4, 1)
slope = coefficients[0]
intercept = coefficients[1]


# Generate the best-fit line
best_fit_line1 = np.polyval(coefficients, years1)
best_fit_line2 = np.polyval(coefficients2, years1)
best_fit_line3 = np.polyval(coefficients3, years1)
best_fit_line4 = np.polyval(coefficients4, years1)


# Plot the data and best-fit line
plt.scatter(years1, data1, color='b', label='Data')
plt.scatter(years1, data2, color='b', label='Data2')
plt.scatter(years1, data3, color='b', label='Data3')
plt.scatter(years1, data4, color='b', label='Data4')


plt.plot(years1, best_fit_line1, color='r', label='Best Fit Line')
plt.plot(years1, best_fit_line2, color='r', label='Best Fit Line')
plt.plot(years1, best_fit_line3, color='r', label='Best Fit Line')
plt.plot(years1, best_fit_line4, color='r', label='Best Fit Line')


# Display the interactive plot
ax.legend()

# Generate your interactive graph using Dash components
graph_component = dcc.Graph(...)

# Convert the figure to an interactive HTML representation
html_fig = mpld3.fig_to_html(fig)

# Save the HTML to a file
with open('figure.html', 'w') as f:
    f.write(html_fig)

plt.show()



