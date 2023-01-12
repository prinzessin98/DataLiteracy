#import math
#import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

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
#print(df.head())
#introduce weeks as new time index to sum up days
weeks = pd.date_range('2018-01-01', '2022-11-27', freq='W').to_numpy()
weeks= np.repeat(weeks, 7)
#reshape the dataframe to show the sum of all cyclists of a week
df['week']= weeks.tolist()
df['week']=pd.to_datetime(df['week']).dt.date
df=df.groupby('week').sum()
#print(df.tail())
#plotten
#plt.plot(raw_df)
#plt.plot(df)
#plt.show()
df.index=pd.to_datetime(df.index)
############# Parameters ######################

start_date_before="-01-01"
end_date_before="-04-05"
start_date_after="-06-04"
end_date_after="-04-05"

############# Train data Part 1 #########################

#dates18=pd.date_range("2018"+start_date_before, "2018"+end_date_before, freq='W')
#dates19=pd.date_range("2019"+start_date_before, "2019"+end_date_before, freq='W')
#dates20=pd.date_range("2020"+start_date_before, "2020"+end_date_before, freq='W')
#dates21=pd.date_range("2021"+start_date_before, "2021"+end_date_before, freq='W')
dates=pd.date_range("2022"+start_date_after, "2022-11-27", freq='W')
df_train=df.copy()
df_train.index=pd.date_range("2018"+start_date_before, "2022-11-27", freq='W')
df_train=df_train.drop(dates)
print(df_train.tail())

##print(pd.to_datetime(df.index))
##print(y_train)
##plot tunnel for each week separately over years
#f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True)
#for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
#    ax1.plot([2018,2019,2020,2021],df['Tunnel'].filter(items=[d18,d19,d20,d21],axis=0))
#ax1.set_title('Tunnel')
##plt.plot([2018,2019,2020,2021],df.filter(items=[dates18[0],dates19[0],dates20[0],dates21[0]],axis=0))
##plt.show()
##plot Steinlach for each week separately over years
#for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
#    ax2.plot([2018,2019,2020,2021],df['Steinlach'].filter(items=[d18,d19,d20,d21],axis=0))
#ax2.set_title('Steinlach')
##plt.show()
##plot Hirschau for each week separately over years
#for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
#    ax3.plot([2018,2019,2020,2021],df['Hirschau'].filter(items=[d18,d19,d20,d21],axis=0))
#ax3.set_title('Hirschau')

##plt.show()
############## Train data Part 2 #########################

#dates18=pd.date_range("2018"+start_date_after, "2018"+end_date_after, freq='W')
#dates19=pd.date_range("2019"+start_date_after, "2019"+end_date_after, freq='W')
#dates20=pd.date_range("2020"+start_date_after, "2020"+end_date_after, freq='W')
#dates21=pd.date_range("2021"+start_date_after, "2021"+end_date_after, freq='W')
#dates=dates18.append(dates19).append(dates20).append(dates21)
##print(pd.to_datetime(df.index))
##print(y_train)
##plot tunnel for each week separately over years
#f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True)

#for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
#    ax1.plot([2018,2019,2020,2021],df['Tunnel'].filter(items=[d18,d19,d20,d21],axis=0))
##plt.plot([2018,2019,2020,2021],df.filter(items=[dates18[0],dates19[0],dates20[0],dates21[0]],axis=0))
#ax1.set_title('Tunnel')
##plot Steinlach for each week separately over years
#for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
#    ax2.plot([2018,2019,2020,2021],df['Steinlach'].filter(items=[d18,d19,d20,d21],axis=0))
#ax2.set_title('Steinlach')
##plot Hirschau for each week separately over years
#for d18,d19,d20,d21 in zip(list(dates18),list(dates19),list(dates20),list(dates21)):
#    ax3.plot([2018,2019,2020,2021],df['Hirschau'].filter(items=[d18,d19,d20,d21],axis=0))
#ax3.set_title('Hirschau')
##plt.show()

############# Test data  #########################

dates22_before = pd.date_range("2022"+start_date_before, "2022"+end_date_before, freq='W').date
dates22_after = pd.date_range("2022"+start_date_after, "2022"+end_date_after, freq='W').date
#print(pd.to_datetime(df.index))
#print(y_train)
#plot tunnel for each week separately over years
cyclists22_before= df.filter(items=list(dates22_before),axis=0)
cyclists22_after= df.filter(items=list(dates22_after),axis=0)
#print(len(cyclists22_before),len(dates22_before))

########### Prediction with seasonality #######################

fourier = CalendarFourier(freq="A", order=15)  # 15 sin/cos pairs for "A"nnual seasonality
def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
#plot_periodogram(df["Tunnel"])
plt.show()
dp = DeterministicProcess(
    index=df_train.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index

for loc in ["Tunnel","Steinlach","Hirschau"]:
    fig, a = plt.subplots(1,1)
    y = df_train[loc]
    model = LinearRegression(fit_intercept=False)
    _ = model.fit(X, y)

    y_pred = pd.Series(model.predict(X), index=y.index)
    X_fore = dp.out_of_sample(steps=40)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
    tit=f" {loc} Traffic - Seasonal Forecast"

    a = y.plot(color="blue", style='.', title=tit)
    a = y_pred.plot(ax=a, label="Seasonal")
    a = y_fore.plot(ax=a, label="Seasonal Forecast", color='C3')
    _ = a.legend()

    plt.show()