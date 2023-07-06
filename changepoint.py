"""
Sergio Pérez Pantoja    Matcom C-411
Changepoint detection
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate a time series

np.random.seed(123) # For reproducibility
n = 500
# We use a normal distribution with K* = 2 changes in its params
# mean (1st param) and standard deviation (2nd param) 
ts = np.concatenate(
    [
        np.random.normal(0, 1, 200),
        np.random.normal(3, 1, 200),
        np.random.normal(0, 2, 100)
    ]
)

plt.figure(figsize=(10, 6))
plt.plot(range(n), ts)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Simulated Time Series')
plt.show()


