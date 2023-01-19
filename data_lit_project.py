import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from yellowbrick.regressor import ResidualsPlot, PredictionError



register_matplotlib_converters()
##Loading the data
p = Path.cwd()
p = str(p)+"/Daten/data.csv"
raw_df =  pd.read_csv(p, header=1,sep=";", names=['date','Tunnel','Steinlach','Hirschau'])
#sum up cyclists over each day
raw_df.head
raw_df['date']=pd.to_datetime(raw_df['date']).dt.date
raw_df=raw_df.groupby(['date']).sum()
df=raw_df.copy()
#introduce weeks as new time index to sum up days
weeks = pd.date_range('2018-01-01', '2022-11-27', freq='W').to_numpy()
weeks= np.repeat(weeks, 7)
#reshape the dataframe to show the sum of all cyclists of a week
df['week']= weeks.tolist()
df['week']=pd.to_datetime(df['week']).dt.date
df=df.groupby('week').sum()
df.index=pd.to_datetime(df.index)

############# Parameters ######################

start_date_before="-01-01"
end_date_before="-04-05"
end_date_train="-12-31"
start_date_test="-01-01"
start_date_after="-04-06"
end_date_after="-11-27"

############# Train data #########################

dates_after=pd.date_range("2022"+start_date_test, "2022"+end_date_after, freq='W')
df_train=df.copy()
df_train.index=pd.date_range("2018"+start_date_before, "2022"+end_date_after, freq='W')
df_train=df_train.drop(dates_after)
print(df_train.tail())


############# Test data  #########################

dates22_before = pd.date_range("2022"+start_date_test, "2022"+end_date_before, freq='W').date
dates22_after = pd.date_range("2022"+start_date_after, "2022"+end_date_after, freq='W').date
#plot tunnel for each week separately over years
if (not isinstance(df.index, np.ndarray)):
    df.index = df.index.date 
cyclists22= df.filter(items=pd.date_range("2022"+start_date_test, "2022"+end_date_after, freq='W').date)

cyclists22_before= df.filter(items=list(dates22_before),axis=0)
cyclists22_after= df.filter(items=list(dates22_after),axis=0)

########### Prediction with seasonality #######################

#number of fourier pairs
fourierno=28
fourier = CalendarFourier(freq="A", order=fourierno)  # sin/cos pairs for Annual seasonality

# function to plot a periodogram
def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1W")
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
    ax.set_xticks([1, 2, 4, 6, 12, 26])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)"
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
#plot_periodogram(df["Tunnel"])
#plt.show()

#fitiing fourier pairs to data
dp = DeterministicProcess(
    index=df_train.index,
    constant=True,               
    order=1,                     
    seasonal=False,               
    additional_terms=[fourier],
    drop=True)

X = dp.in_sample()  # create features for dates in tunnel.index

# create plots for counting points predicting the no of cyclists for 2022 for each measure point or sum
only_sum=True
if only_sum:
    loc="Tübingen"
    fig, a = plt.subplots(1,1)
    y = df_train.sum(axis=1)
    model = LinearRegression(fit_intercept=False)
    _ = model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index) 
    X_fore = dp.out_of_sample(48)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
    tit=f" {loc} Cycle Traffic - Seasonal Forecast"
    cyclists22_before.sum(axis=1).plot(color="red",style='^', label="Test data bef. blocking")
    cyclists22_after.sum(axis=1).plot(color="red",style='*', label="Test data aft. blocking")
    a.vlines(dates22_after[0], df.sum(axis=1).min(), df.sum(axis=1).max(),color="darkgrey", linestyles="dashed", label="Mühlstraße blocked for cars")    
    a = y.plot(color="blue", style='.', label="Train data", title=tit)
    a = y_pred.plot(ax=a, label="Train pred.")
    a = y_fore.plot(ax=a, label="Test pred.", color='red')
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.grid(False)
    plt.xlabel("date")
    plt.ylabel("No. of cyclists/week")
    _ = a.legend(ncol=2,loc="lower left")
    plt.tight_layout()
    #   print(y_fore)
    #plt.show()
    plt.savefig(f"{loc}_pred_fourierno{fourierno}.png",dpi=600)

else:
    for loc in ["Tunnel","Steinlach","Hirschau"]:
        fig, a = plt.subplots(1,1)
        y = df_train[loc]
        model = LinearRegression(fit_intercept=False)
        _ = model.fit(X, y)

        y_pred = pd.Series(model.predict(X), index=y.index) 
        X_fore = dp.out_of_sample(48)
        y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
        tit=f" {loc} Traffic - Seasonal Forecast"
        cyclists22_before[loc].plot(color="red",style='^', label="Test data bef. blocking")
        cyclists22_after[loc].plot(color="red",style='*', label="Test data aft. blocking")
        a.vlines(dates22_after[0], df[loc].min(), df[loc].max(),color="darkgrey", linestyles="dashed", label="Mühlstraße blocked for cars")    
        a = y.plot(color="blue", style='.', label="Train data", title=tit)
        a = y_pred.plot(ax=a, label="Train pred.")
        a = y_fore.plot(ax=a, label="Test pred.", color='red')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.grid(False)
        plt.xlabel("date")
        plt.ylabel("No. of cyclists/week")
        _ = a.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"{loc}_pred_fourierno{fourierno}.png",dpi=600)
# create plots for counting points predicting the no of cyclists for 2022 for sum

    
for loc in ["Tunnel","Steinlach","Hirschau"]:
    y = df_train[loc]
    model = LinearRegression(fit_intercept=False)
    _ = model.fit(X, y)
    X_fore = dp.out_of_sample(len(cyclists22_after))

    # plot the residuals training data
    visualizer = ResidualsPlot(model, qqplot=True, hist=False)  
    visualizer.fit(X, y)  # Fit the training data to the visualizer
    visualizer.score(X_fore, cyclists22_after[loc])  # Evaluate the model on the test data
    visualizer.show()  # Finalize and render the figure
    
    # plot the prediction error
    pred_err_train = PredictionError(model)
    pred_err_train.fit(X,y)    #prediction error for train set
    pred_err_train.score(X,y)
    pred_err_train.show() 
    
    pred_err_test = PredictionError(model)
    pred_err_test.fit(X,y)                             
    pred_err_test.score(X_fore, cyclists22_after[loc]) # prediction error for test set
    pred_err_test.show()
