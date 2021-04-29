# This program is the python code used in the Programming and Scripting 2021 Project
# The aim of this code is to carry out data analysis on the Fisher's Irish data set

# Author: Fionn McCarthy - G00301126.

import numpy as np # wused for working with arrays 
import pandas as pd # used for importing data for data analysis and manipulating data
import matplotlib.pyplot as plt # used for data visualisation and plotting
import seaborn as sns # used for visualisations and ivestigating random distributions

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


font1 = {'family':'serif','color':'darkblue','size':14} 
font2 = {'family':'serif','color':'lightblue','size':12} 


#For plotting the data I will group it into each flower class
iris_virginica = irisdata[irisdata["class"] == "Iris-virginica"] # [12]
iris_versicolor = irisdata[irisdata["class"] == "Iris-versicolor"] # [12]
iris_setosa = irisdata[irisdata["class"] == "Iris-setosa"]  # [12]

sns.set(style="darkgrid") # sns.set_theme sts the backround colour of the grid

# Histograms of Data
# https://datavizpyr.com/overlapping-histograms-with-matplotlib-in-python/ [13]
#sepal_length_histogram, axes = plt.subplots(figsize=(10,8))
def histogram_plot(p1, p2, p3): # I Will create a histogram 
    sns.histplot(data = iris_virginica[p1], kde = False, label = 'Iris virginica', color = 'red') # https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn [14]
    sns.histplot(data = iris_versicolor[p1], kde = False, label = 'Iris versicolor', color = 'skyblue')
    sns.histplot(data = iris_setosa[p1],  kde = False, label = 'Iris setosa', color = 'yellow')
    plt.xlabel(p2, fontdict = font2)
    plt.ylabel("Frequency", fontdict = font2)
    plt.title("Histogram Plot of " + p2 + " by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper right')
    plt.savefig(p3)
    plt.show()

def histograms():
    histogram_plot('sepallength', "Sepal Length", "sepal_length_histogram")
    histogram_plot('sepalwidth', "Sepal Width", "sepal_width_histogram")
    histogram_plot('petallength', "Petal Length", "petal_length_histogram")
    histogram_plot('petalwidth', "Petal Width", "petal_width_histogram")



# https://seaborn.pydata.org/generated/seaborn.scatterplot.html [15]
def scatterplot_petal(p1, p2, p3, p4, p5):
    sns.scatterplot(data = irisdata, x = p1, y = p2, hue = "class", palette = "deep")
    plt.xlabel(p3, fontdict = font2)
    plt.ylabel(p4, fontdict = font2)
    plt.title("Scatterplot of " + p3 +" and " + p4 + " by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper left')
    plt.savefig(p5)
    plt.show()

def scatterplots():
    scatterplot_petal("petallength", "petalwidth", "Petal Length", "Petal Width", "scatterplot_petal")
    scatterplot_petal("sepallength", "sepalwidth", "Sepal Length", "Sepal Width", "scatterplot_sepal")
    scatterplot_petal("petallength", "sepallength", "Petal Length", "Sepal Length", "scatterplot_petallength_sepallength")
    scatterplot_petal("petalwidth", "sepalwidth", "Petal Width", "Sepal Width", "scatterplot_petalwidth_sepalwidth")



# Pair plots were used in order to compare scatterplots of all teh possibloe varaibles
# https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166 [16]
def iris_pairplot():
    sns.pairplot(data = irisdata, hue = "class", palette = "deep", kind = "scatter")
    plt.savefig("iris_pairplot")
    plt.show()

#Executing the functions
histograms()
scatterplots()
iris_pairplot()

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
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html grouping the variables [12] 24/04/2021
# https://datavizpyr.com/overlapping-histograms-with-matplotlib-in-python/  [13]
# https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn [14]
# https://seaborn.pydata.org/generated/seaborn.scatterplot.html [15]
# https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166 [16]