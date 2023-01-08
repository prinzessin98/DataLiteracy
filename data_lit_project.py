#import math
#import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
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
print(df.head())
#plotten
#plt.plot(raw_df)
plt.plot(df)
plt.show()