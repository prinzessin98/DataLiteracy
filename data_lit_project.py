#import math
#import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression as LinReg
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
##Loading the data
#path ="C:/Users/Fynn/Documents/Uni/Master/DataLiteracy/Project/Daten/data.csv"
p = Path.cwd()
p = str(p)+"/Daten/data.csv"
raw_df =  pd.read_csv(p, header=1,sep=";", names=['date','Tunnel','Steinlach','Hirschau'])
#sum up cyclists over each day
raw_df.head
raw_df['date']=pd.to_datetime(raw_df['date']).dt.date
raw_df=raw_df.groupby(['date']).sum()
df=raw_df.copy()
print(df.head())
#introduce weeks as new time index to sum up days
weeks = pd.date_range('2018-01-01', '2022-11-27', freq='W').to_numpy()
weeks= np.repeat(weeks, 7)
#reshape the dataframe to show the sum of all cyclists of a week
df['week']= weeks.tolist()
df['week']=pd.to_datetime(df['week']).dt.date
df=df.groupby('week').sum()
print(df.tail())
#plotten
#plt.plot(raw_df)
#plt.plot(df)
#plt.show()
############# Train data Part 1 #########################

dates18=pd.date_range('2018-01-01', '2018-04-05', freq='W')
dates19=pd.date_range('2019-01-01', '2019-04-05', freq='W')
dates20=pd.date_range('2020-01-01', '2020-04-05', freq='W')
dates21=pd.date_range('2021-01-01', '2021-04-05', freq='W')
dates=dates18.append(dates19).append(dates20).append(dates21)
#print(pd.to_datetime(df.index))
df.index=pd.to_datetime(df.index)
#print(y_train)
#plot tunnel for each week separately over years
for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
    plt.plot([2018,2019,2020,2021],df['Tunnel'].filter(items=[d18,d19,d20,d21],axis=0))
#plt.plot([2018,2019,2020,2021],df.filter(items=[dates18[0],dates19[0],dates20[0],dates21[0]],axis=0))
plt.show()
#plot Steinlach for each week separately over years
for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
    plt.plot([2018,2019,2020,2021],df['Steinlach'].filter(items=[d18,d19,d20,d21],axis=0))
plt.show()
#plot Hirschau for each week separately over years
for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
    plt.plot([2018,2019,2020,2021],df['Hirschau'].filter(items=[d18,d19,d20,d21],axis=0))
plt.show()

#reg = LinReg().fit(X_train,y_train)
