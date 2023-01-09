#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#1000 rows

visa = pd.read_csv('/Users/elizabethwellington/Desktop/VA FINAL PROJECT/lattttt na removed .csv')
visa

plt.scatter(visa['lon'],visa['lat'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

cluss = KMeans(n_clusters=4)
cluss

predict1 = cluss.fit_predict(visa[['lon','lat']])
predict1

visa['Cluster']=predict1
visa

#cluster plots


cluss.cluster_centers_
#kmeans clustering for year 2016

df1 = visa[visa.Cluster==0]
df2 = visa[visa.Cluster==1]
df3 = visa[visa.Cluster==2]
df4 = visa[visa.Cluster==3]
plt.scatter(df1.lon,df1['lat'],color='green')
plt.scatter(df2.lon,df2['lat'],color='red')
plt.scatter(df3.lon,df3['lat'],color='black')
plt.scatter(df4.lon,df4['lat'],color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#500 rows 

visaa = pd.read_csv('/Users/elizabethwellington/Desktop/VA FINAL PROJECT/lattt vs longgggg.csv')
visaa

plt.scatter(visaa['lon'],visaa['lat'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

clus = KMeans(n_clusters=4)
clus

predict = clus.fit_predict(visaa[['lon','lat']])
predict

visaa['Cluster']=predict
visaa

#cluster plots


clus.cluster_centers_
#kmeans clustering for year 2016

df1 = visaa[visaa.Cluster==0]
df2 = visaa[visaa.Cluster==1]
df3 = visaa[visaa.Cluster==2]
df4 = visaa[visaa.Cluster==3]
plt.scatter(df1.lon,df1['lat'],color='green')
plt.scatter(df2.lon,df2['lat'],color='red')
plt.scatter(df3.lon,df3['lat'],color='black')
plt.scatter(df4.lon,df4['lat'],color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#k means clustering for year 2015

visa15 = pd.read_csv('/Users/elizabethwellington/Desktop/VA FINAL PROJECT/lat long 2015 1000.csv')
visa15

plt.scatter(visa15['lon'],visa15['lat'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

clus = KMeans(n_clusters=4)
clus

predict = clus.fit_predict(visaa[['lon','lat']])
predict

visa15['Cluster']=predict
visa15

#cluster plots


clus.cluster_centers_
#kmeans clustering for year 2016

df1 = visa15[visa15.Cluster==0]
df2 = visa15[visa15.Cluster==1]
df3 = visa15[visa15.Cluster==2]
df4 = visa15[visa15.Cluster==3]
plt.scatter(df1.lon,df1['lat'],color='green')
plt.scatter(df2.lon,df2['lat'],color='red')
plt.scatter(df3.lon,df3['lat'],color='black')
plt.scatter(df4.lon,df4['lat'],color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#job type for the year 2016 
visaa15 = pd.read_csv('/Users/elizabethwellington/Desktop/VA FINAL PROJECT/long lat na removed 2015.csv')
visaa15

plt.scatter(visaa15['lon'],visaa15['lat'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

cluss1 = KMeans(n_clusters=4)
cluss1

predict2 = cluss1.fit_predict(visa[['lon','lat']])
predict2

visaa15['Cluster']=predict2
visaa15

#cluster plots


cluss.cluster_centers_
#kmeans clustering for year 2016

df1 = visaa15[visaa15.Cluster==0]
df2 = visaa15[visaa15.Cluster==1]
df3 = visaa15[visaa15.Cluster==2]
df4 = visaa15[visaa15.Cluster==3]
plt.scatter(df1.lon,df1['lat'],color='green')
plt.scatter(df2.lon,df2['lat'],color='red')
plt.scatter(df3.lon,df3['lat'],color='black')
plt.scatter(df4.lon,df4['lat'],color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

