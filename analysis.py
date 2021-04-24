# This program is the python code used in the Programming and Scripting 2021 Project
# The aim of this code is to carry out data analysis on the Fisher's Irish data set

# Author: Fionn McCarthy - G00301126.

import numpy as np # wused for working with arrays 
import pandas as pd # used for importing data for data analysis

irisdata = pd.read_csv('iris_data.csv')
pd.set_option("display.precision",2) # set figures so be returned to two decimal places

iris_summary_file = open("iris_summary.txt","w") # Opened and create dtext file iris_summary.txt for writing 
print("This file is displaying the summary statistics for the Iris dataset.", file = iris_summary_file)
print ("\n", file = iris_summary_file) #line break
print("Below shows how the Fisher's Iris dataset looks:", file = iris_summary_file)
print(irisdata, file = iris_summary_file)
print ("\n", file = iris_summary_file) #line break
print("The describe() function shows the summary statistics of the numeric data below:", file = iris_summary_file)
print(irisdata.describe(), file = iris_summary_file) # descibe shows count, mean, standard deviation, minimum, median, interquartile range [2]
print ("\n", file = iris_summary_file) #line break
print("The statistics on the non-numeric (flower class) data can be seen below:", file = iris_summary_file)
print(irisdata.describe(include=['object', 'bool']), file = iris_summary_file) # shows us statistics on non-numerical values [2]
print ("\n", file = iris_summary_file) #line break
print("The number of records for each class can be seen below:", file = iris_summary_file)
print(irisdata['class'].value_counts(), file = iris_summary_file) # number of samples for each type of iris flower
print ("\n", file = iris_summary_file) #line break
print("Below shows a summary of each of the variables grouped by the flower class:", file = iris_summary_file)
print(irisdata.groupby(['class']).describe(), file = iris_summary_file) # [3] group data by flower class and summarize

#iris_sepallength = irisdata.groupby(['class'])[['sepallength']].describe()
#print(iris_sepallength)
#iris_sepalwidth = irisdata.groupby(['class'])[['sepalwidth']].describe()
#print(iris_sepalwidth)
#iris_petallength = irisdata.groupby(['class'])[['petallength']].describe()
#print(iris_petallength)
#iris_petalwidth = irisdata.groupby(['class'])[['petalwidth']].describe()
#print(iris_petalwidth)

# https://datahub.io/machine-learning/iris data set source on 17/04/2021
# https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas [2] pandas 19/04/2021
# https://www.earthdatascience.org/courses/intro-to-earth-data-science/scientific-data-structures-python/pandas-dataframes/run-calculations-summary-statistics-pandas-dataframes/ [3] 20/04/2021
# https://medium.com/@avulurivenkatasaireddy/exploratory-data-analysis-of-iris-data-set-using-python-823e54110d2d [4] 21/04/2021
# https://medium.com/@Nivitus./iris-flower-classification-machine-learning-d4e337140fa4 [5] image source 21/04/2021
# https://www.jstor.org/stable/2394164?origin=crossref&seq=1 [6] Who collated the data 17/04/2021 
# https://en.wikipedia.org/wiki/Ronald_Fisher [7]
# https://en.wikipedia.org/wiki/Iris_flower_data_set Iris Dataset 17/04/2021 [8]
# https://www.w3schools.com/python/numpy/numpy_intro.asp numpy 17/04/2021 [9]
# https://www.activestate.com/resources/quick-reads/what-is-matplotlib-in-python-how-to-use-it-for-plotting/ matplotlib [10] 17/04/2021
# https://seaborn.pydata.org/ seaborn [11] 17/04/2021
