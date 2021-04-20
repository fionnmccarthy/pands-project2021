# This program is the python code used in the Programming and Scripting 2021 Project
# The aim of this code is to carry out data analysis on the Fisher's Irish data set

# Author: Fionn McCarthy - G00301126.

import numpy as np # wused for working with arrays 
import pandas as pd # used for importing data for data analysis

irisdata = pd.read_csv('iris_data.csv')
pd.set_option("display.precision",2) # set figures so be returned to two decimal places

print(irisdata.head()) # first 5 lines of the data [2]

print(irisdata.shape) # number rows and colums in the data [2]

print(irisdata.columns) # printing out the column names  [2]

print(irisdata.info()) # this prints out general info about the data [2]

print(irisdata.describe()) # descibe shows count, mean, standard deviation, minimum, median, interquartile range [2]

print(irisdata.describe(include=['object', 'bool'])) # shows us statistics on non-numerical values [2]


iris_virginica = irisdata.groupby(['class']).describe() # [3] group data by flower class and summarize
print(iris_virginica)
# iris_sepalwidth = irisdata.groupby(['class'])[['sepalwidth']].describe() 

#print(irisdata)

# https://datahub.io/machine-learning/iris data set source on 17/04/2021
# https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas [2] pandas 19/04/2021
# https://www.earthdatascience.org/courses/intro-to-earth-data-science/scientific-data-structures-python/pandas-dataframes/run-calculations-summary-statistics-pandas-dataframes/ [3] 20/04/2021