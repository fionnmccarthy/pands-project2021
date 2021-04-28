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


font1 = {'family':'serif','color':'darkblue','size':15} 
font2 = {'family':'serif','color':'darkred','size':12} 


#For plotting the data I will group it into each flower class
iris_virginica = irisdata[irisdata["class"] == "Iris-virginica"] # [12]
iris_versicolor = irisdata[irisdata["class"] == "Iris-versicolor"] # [12]
iris_setosa = irisdata[irisdata["class"] == "Iris-setosa"]  # [12]

sns.set(style="darkgrid") # sns.set_theme sts the backround colour of the grid
# Histograms of Data
# https://datavizpyr.com/overlapping-histograms-with-matplotlib-in-python/ [13]
#sepal_length_histogram, axes = plt.subplots(figsize=(10,8))
def histogram_sepal_length():
    sns.histplot(data = iris_virginica['sepallength'], kde = False, label = 'Iris virginica', color = 'red') # https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn [14]
    sns.histplot(data = iris_versicolor['sepallength'], kde = False, label = 'Iris versicolor', color = 'skyblue')
    sns.histplot(data = iris_setosa['sepallength'],  kde = False, label = 'Iris setosa', color = 'yellow')
    plt.xlabel("Sepal Length (cm)", fontdict = font2)
    plt.ylabel("Frequency", fontdict = font2)
    plt.title("Histogram Plot of Sepal Length by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper right')
    plt.savefig("sepal_length_histogram")
    plt.show()

def histogram_sepal_width():
    sns.histplot(data = iris_virginica['sepalwidth'], kde = False, label = 'Iris virginica', color = 'red') # https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn [14]
    sns.histplot(data = iris_versicolor['sepalwidth'], kde = False, label = 'Iris versicolor', color = 'skyblue')
    sns.histplot(data = iris_setosa['sepalwidth'],  kde = False, label = 'Iris setosa', color = 'yellow')
    plt.xlabel("Sepal Width (cm)", fontdict = font2)
    plt.ylabel("Frequency", fontdict = font2)
    plt.title("Histogram Plot of Sepal Width by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper right')
    plt.savefig("sepal_width_histogram")
    plt.show()

def histogram_petal_length(): 
    sns.histplot(data = iris_virginica['petallength'], kde = False, label = 'Iris virginica', color = 'red') # https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn [14]
    sns.histplot(data = iris_versicolor['petallength'], kde = False, label = 'Iris versicolor', color = 'skyblue')
    sns.histplot(data = iris_setosa['petallength'],  kde = False, label = 'Iris setosa', color = 'yellow')
    plt.xlabel("Petal Length (cm)", fontdict = font2)
    plt.ylabel("Frequency", fontdict = font2)
    plt.title("Histogram Plot of Petal Length by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper right')
    plt.savefig("petal_length_histogram")
    plt.show()

def histogram_petal_width():
    sns.histplot(data = iris_virginica['petalwidth'], kde = False, label = 'Iris virginica', color = 'red') # https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn [14]
    sns.histplot(data = iris_versicolor['petalwidth'], kde = False, label = 'Iris versicolor', color = 'skyblue')
    sns.histplot(data = iris_setosa['petalwidth'],  kde = False, label = 'Iris setosa', color = 'yellow')
    plt.xlabel("Petal Width (cm)", fontdict = font2)
    plt.ylabel("Frequency", fontdict = font2)
    plt.title("Histogram Plot of Petal Width by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper right')
    plt.savefig("petal_width_histogram")
    plt.show()

# executing the fuctions to plot the scatterplots
histogram_sepal_length() 
histogram_sepal_width() 
histogram_petal_length()
histogram_petal_width()

# https://seaborn.pydata.org/generated/seaborn.scatterplot.html [15]
def scatterplot_petal():
    sns.scatterplot(data = irisdata, x = "petallength", y = "petalwidth", hue = "class", palette = "deep")
    plt.xlabel("Petal Length (cm)", fontdict = font2)
    plt.ylabel("Petal Width (cm)", fontdict = font2)
    plt.title("Scatterplot of Petal Length and Petal Width by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper left')
    plt.savefig("scatterplot_petal")
    plt.show()

def scatterplot_sepal():
    sns.scatterplot(data = irisdata, x = "sepallength", y = "sepalwidth", hue = "class", palette = "deep")
    plt.xlabel("Sepal Length (cm)", fontdict = font2)
    plt.ylabel("Sepal Width (cm)", fontdict = font2)
    plt.title("Scatterplot of Sepal Length and Sepal Width by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper left')
    plt.savefig("scatterplot_sepal")
    plt.show()

def scatterplot_petallength_sepallength():
    sns.scatterplot(data = irisdata, x = "petallength", y = "sepallength", hue = "class", palette = "deep")
    plt.xlabel("Petal Length (cm)", fontdict = font2)
    plt.ylabel("Sepal Length (cm)", fontdict = font2)
    plt.title("Scatterplot of Petal Length and Sepal Length by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper left')
    plt.savefig("scatterplot_petallength_sepallength")
    plt.show()

def scatterplot_petalwidth_sepalwidth():
    sns.scatterplot(data = irisdata, x = "petalwidth", y = "sepalwidth", hue = "class", palette = "deep")
    plt.xlabel("Petal Width (cm)", fontdict = font2)
    plt.ylabel("Sepal Width (cm)", fontdict = font2)
    plt.title("Scatterplot of Petal Width and Sepal Width by Flower Species", fontdict = font1) 
    plt.grid(color = 'grey', ls = '--', lw = 0.5) 
    plt.legend(loc='upper right')
    plt.savefig("scatterplot_petalwidth_sepalwidth")
    plt.show()

# executing the fuctions to plot the scatterplots
scatterplot_petal()
scatterplot_sepal()
scatterplot_petallength_sepallength()
scatterplot_petalwidth_sepalwidth()

# Pair plots were used in order to compare scatterplots of all teh possibloe varaibles
# https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166 [16]
def iris_pairplot():
    sns.pairplot(data = irisdata, hue = "class", palette = "deep", kind = "scatter")
    plt.savefig("iris_pairplot")
    plt.show()

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