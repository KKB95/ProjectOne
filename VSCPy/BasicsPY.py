import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import pprint as pp

# New Commit
###############################################################################################

# Importsnt Notes:

# All variable names in Python are said to be references to the values
# An is expression evaluates to True if two variables point to the same (identical) object.

a = 254
b = 254
a is b
# Answer : True

a = 257
b = 257
a is b
# Answer : False

###############################################################################################

# Plotting in Python
X = [1, 2, 3, 4, 5, 6, 7, 8]
F = [1, 2, 3, 4, 5, 6, 7, 8]
T = [1, 2, 4, 8, 16, 32, 64, 128]
A = [100, 50, 25, 12.5, 6.25, 3.125, 1.5625, 0.78125]
FR = [100, 25, 6.25, 1.5625, 0.39, 0.09, 0.02, 0.006]

df = pd.DataFrame({'X': X, 'F': F, 'T': T, 'A': A, 'FR': FR})

# plt.plot('X', 'F', data=df, marker='', color='skyblue', linewidth=2)
plt.plot('X', 'T', data=df, marker='', color='skyblue', linewidth=2)
plt.plot('X', 'A', data=df, marker='', color='red', linewidth=2)
plt.plot('X', 'FR', data=df, marker='', color='black', linewidth=2)
plt.axis([0, 8, 0, 100])
plt.show()

###############################################################################################

# List/Dictionaries in python

# Lists:
L1 = ['Bobby Sam', 36, 30000, 'Database']
L2 = ['Ian Ruby', 38, 40000, 'Software']

employeeList = [L1, L2]
# fetching details

# 1
L1[0], L2[1]

# 2
for person in employeeList:
    print(person)

# 3
employeeList[1][0]

# Using Maps & Filters

dict_a = [{'name': 'python', 'points': 10}, {'name': 'java', 'points': 8}]
list_a = [1, 2, 3, 0]
list_b = [10, 11, 12, 14]

# Maps

# Note: Lists have to be of the same size
newList = map(lambda x, y: x + y, list_a, list_b)
list(newList)

names = map(lambda x: x['name'], dict_a)
list(names)

pointearned = map(lambda x: x['points']*10,  dict_a)
list(pointearned)
condition = map(lambda x: x['name'] == "python", dict_a)
list(condition)

# Filters
filterList = filter(lambda x: x % 2 != 0, list_a)
list(filterList)

###############################################################################################


def func(B=None, C=None, D=None, E=None):
    ss = "Borrow"
    sd = "Sorrow"
    if (ss == "Borrow"):
        X = B
        Y = C
    if (sd == "Sorrow"):
        Z = D
        F = E
    print(X, Y, Z, F)


def funcOne():
    func(B=3, C=4, D=5, E=6)

###############################################################################################
