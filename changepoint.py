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


# 2. Cost function

def segment_cost(x):
    n = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    logL = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma**2) - 1/(2*sigma**2) * np.sum((x - mu)**2)
    return -logL # We return the negative log-likelihood as we want to minimize the cost


# The choice of the negative log-likelihood as a cost function is justified by 
# the fact that it is a standard approach in statistical model fitting, as 
# maximizing the log-likelihood is equivalent to minimizing the negative 
# log-likelihood. This approach is particularly suitable for data that is 
# thought to follow a Normal distribution with variable mean and standard 
# deviation, as is the case here.


# 3. Optimal partition algorithm

def optimal_partition(signal, seg_cost, K):
    n = len(signal)
    M = np.zeros((K, n, n))

    for u in range(n):
        for v in range(u+1, n):
            M[0][u][v] = segment_cost(signal[u:v+1])
    
    if K > 1:
        for k in range(1, K):
            for u in range(n-k-1):
                for v in range(u+k+1, n):
                    if v-u > k+1:
                        M[k][u][v] = min([M[k-1][u][t] + M[0][t+1][v] for t in range(u+k-1, v)])
    
    L = np.zeros((K+1), dtype=int)
    L[K] = n - 1
    k = K
    while k > 0:
        s = L[k]
        tstar = np.argmin([M[0][t+1][s] + M[k-1][0][t] for t in range(k, s)])
        L[k-1] = tstar
        k -= 1
    return L[:-1]


cp = optimal_partition(ts, segment_cost, 2)
print(cp)
#plot
plt.figure(figsize=(10, 6))
plt.plot(range(n), ts)
for c in cp:
    plt.axvline(c, color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Optimal Partition')
plt.show()


# 4. PELT algorithm

def PELT(data, seg_cost, pen, n_min):
    if pen is None:
        pen = np.log(len(data))

    F = np.zeros(len(data), np.float32)
    R = np.array([0], dtype=np.int32)
    CP = np.zeros(len(data), np.int32)
    F[:n_min] = -pen

    for tstar in range(n_min, len(data)):
        cost = np.zeros(R.shape[0], np.float32)
        parr = []                  

        for i, t in enumerate(R):
            cost[i] = F[t] + seg_cost(data[t:tstar+1])
            if abs(t - tstar) >= n_min:     
                parr.append((cost[i] + pen, t))
        if len(parr) > 0:
            F[tstar], CP[tstar] = min(parr)
        else:
            continue

        Rstar = []
        for i, c in enumerate(cost):
            if c <= F[tstar]:
                Rstar.append(R[i])
        Rstar.append(tstar)
        R = np.array(Rstar, dtype=np.int32)

    changepoints = [CP[-1]]
    while changepoints[-1] > 0:
        changepoints.append(CP[changepoints[-1]])
    changepoints = changepoints[::-1]
    return changepoints

pen = np.log(n)  # penalty value
cp = PELT(ts, segment_cost, pen, 1)
print(cp)
#plot
plt.figure(figsize=(10, 6))
plt.plot(range(n), ts)
for i in range(1, len(cp)):
    plt.axvline(x=cp[i], color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('PELT penalty = log(n)')
plt.show()
        

# 5. Changing penalty values

pen2 = 0.1 * np.log(n)  # low penalty value
cp = PELT(ts, segment_cost, pen2, 1)
print(cp)
#plot
plt.figure(figsize=(10, 6))
plt.plot(range(n), ts)
for i in range(1, len(cp)):
    plt.axvline(x=cp[i], color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('PELT penalty = 0.1 * log(n)')
plt.show()

pen3 = 2 * np.log(n)  # high penalty value
cp = PELT(ts, segment_cost, pen3, 1)
print(cp)
#plot
plt.figure(figsize=(10, 6))
plt.plot(range(n), ts)
for i in range(1, len(cp)):
    plt.axvline(x=cp[i], color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('PELT penalty = 2 * log(n)')
plt.show()

# With a low penalty (e.g., 0.1 * log(n)), the algorithm detects many change points, 
# including some that might be just noise. As the penalty increases, the number 
# of detected change points should decrease, and with a high enough penalty 
# (e.g., 2 * log(n)), the algorithm might miss some actual change points.
