from sklearn.cluster import KMeans
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
style.use('ggplot')

param = pd.read_csv('world-happiness-report-2021.csv', sep=';', usecols=[
                    'Country name', 'Logged GDP per capita', 'Social support',
                    'Freedom to make life choices', 'Ladder score',
                    'Healthy life expectancy']) #Memilih kolom yang akan digunakan

#Replace comma to dot
param['Logged GDP per capita'] = param['Logged GDP per capita'].str.replace(',','.')
param['Social support'] = param['Social support'].str.replace(',','.')
param['Freedom to make life choices'] = param['Freedom to make life choices'].str.replace(',','.')
param['Ladder score'] = param['Ladder score'].str.replace(',','.')
param['Healthy life expectancy'] = param['Healthy life expectancy'].str.replace(',','.')


country_array = np.array(param.iloc[:, 1:6]) #Wrap data tiap negara ke dalam array

print(country_array)

nCluster = 2 #Banyak cluster yang diinginkan

kmeans = KMeans(n_clusters = nCluster)
kmeans.fit(country_array)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)