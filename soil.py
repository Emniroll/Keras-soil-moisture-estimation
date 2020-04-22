import math
from math import sqrt

import numpy as np
from scipy import stats

y = [17.3508, 14.8306, 18.2516, 16.7890, 16.9154, 16.2759, 14.4632, 20.1448, 14.6726, 14.6034, 15.2408, 16.1969,
     16.5235, 14.0918, 16.4586, 15.5923, 14.8415, 19.2548, 18.5014, 15.2776, 13.3017, 18.7445, 17.3951, 15.5996,
     14.7026, 16.5317, 15.6412, 15.8759, 18.0658, 16.4840, 14.9364, 16.3602, 15.1318, 15.4585, 14.7989, 16.6979,
     15.6288, 20.0054, 14.4178]
t = [18.6, 15.2, 18.6, 19.8, 14.2, 16.7, 15.2, 19.8, 15.7, 15.7, 16.7, 18.6, 16.2, 15.7, 16.2, 14.2, 15.2, 13.5, 19.8,
     15.7, 13.5, 19.8, 19.8, 16.2, 16.7, 16.2, 14.2, 16.2, 16.2, 18.6, 16.7, 15.2, 15.7, 16.7, 15.2, 19.8, 14.2, 16.2,
     13.5]
count = 0
tucount = 0
for i in range(len(y)):
    tu = 100 - t[i]
    tucount += tu
    count += abs(y[i] - t[i])
print(count / tucount)

error = []
for i in range(len(t)):
    tu = 100 - t[i]
    error.append(abs(y[i] - t[i]) / tu)

print("Errors: ", error)
print(error)

squaredError = []
for val in error:
    squaredError.append(val * val)  # target-prediction之差平方

print("Square Error: ", squaredError)

print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE

print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
