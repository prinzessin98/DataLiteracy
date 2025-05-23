import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.stats import ttest_ind




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

def plot_residuals(residuals, name, set_type, label, colors, ymax):
    # histogram
    ax = plt.axes()

    for resid, col in zip(residuals, colors):
        counts, bins = np.histogram(resid, bins = 12)
        
        #fig, ax = plt.subplots()
        plt.hist(bins[:-1], bins, weights=counts, color = col, stacked = True, alpha = 0.6)
        
        plt.ylim([0, ymax])
        plt.title(set_type +' '+ name, size = 18)
        plt.ylabel('count',  size = 15)
        plt.yticks(fontsize = 12) 
        plt.xlabel('residual',  size = 15)
        plt.xticks(fontsize = 15) 
        plt.vlines(np.mean(resid), 0, ymax, color = col)

        # remove grid and axes
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(label, fontsize = 15)
    plt.show()

def plot_real_predicted(y_real, y_pred, name, set_type, label, colors, line_min, line_max):
    # dot plot
    ax = plt.axes()

    for real, pred, col in zip(y_real, y_pred, colors):
        plt.plot(real, pred, 'p', color = col)
        plt.title(set_type +' '+ name, size = 18)
        plt.ylabel('predicted values',  size = 15)
        plt.yticks(fontsize = 12) 
        plt.xlabel('real values',  size = 15)
        plt.xticks(fontsize = 15) 
        
        # remove grid and axes
        plt.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.plot(range(line_min,line_max), range(line_min,line_max), 'black')
    label.insert(len(label), 'perfect prediction')
    plt.legend(label, fontsize = 15)
    plt.show()
    
    
def welch_dof(x,y):
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
    print(f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")
        


col_train = 'blue'
col_control = 'orangered'
col_test = 'orange'

X = dp.in_sample()  # create features for dates in tunnel.index

only_sum=True
if only_sum:
# model and plot over the sum of all three stations
    loc="Tübingen"
    y_train = df_train.sum(axis=1)
    
    #fit the model
    model = LinearRegression(fit_intercept=False)
    _ = model.fit(X, y_train)
    y_train_pred = pd.Series(model.predict(X), index=y_train.index) 

    # make model prediction for 01.01.2022 - 27.11.2022
    X_test = dp.out_of_sample(len(cyclists22_before.sum(axis=1))+ len(cyclists22_after.sum(axis=1)))
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    
    # get the control data (01.01.2022-05.04.2022)
    y_control = cyclists22_before.sum(axis=1)
    y_control_pred = y_pred.filter(pd.date_range("2022"+start_date_before, "2022"+end_date_before, freq='W'))
    
    # get the test data (06.04.2022 -  27.11.2022)
    y_test = cyclists22_after.sum(axis=1)
    y_test_pred = y_pred.filter(pd.date_range("2022"+start_date_after, "2022"+end_date_after, freq='W'))
    
    #### plot the regression plot
    fig, a = plt.subplots(1,1)
    tit=f" {loc} Cycle Traffic - Seasonal Forecast"
    cyclists22_before.sum(axis=1).plot(color=col_control,style='^', label="Test data bef. blocking")
    cyclists22_after.sum(axis=1).plot(color=col_test,style='*', label="Test data aft. blocking")
    a = a.vlines(dates22_after[0], df.sum(axis=1).min(), df.sum(axis=1).max(),color="darkgrey", linestyles="dashed", label="Mühlstraße blocked for cars")    

    a = y_train.plot(color=col_train, style='.', label="Train data", title=tit)
    a = y_train_pred.plot(ax=a, label="Train pred.")
    a = y_test_pred.plot(ax=a, label="Test pred.", color=col_test)
    a = y_control_pred.plot(ax=a, label="Control pred.", color=col_control)

    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.grid(False)
    plt.xlabel("date")
    plt.ylabel("No. of cyclists/week")
    _ = a.legend(ncol=2,loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{loc}_pred_fourierno{fourierno}.png",dpi=600)
    plt.show()
    
    ####### plot residuals in histogram
    # 1) train data
    resid_train = y_train.values - y_train_pred.values
    plot_residuals([resid_train], '(' + str(loc) + ')', 'trainset', ['train set'], [col_train], 50)
    
    # 2) control vs test
    resid_control = y_control.values - y_control_pred.values
    resid_test = y_test.values - y_test_pred.values
    plot_residuals([resid_control, resid_test], '(' + str(loc) + ')', 'control vs test', 
                   ['control set', 'test set'], [col_control, col_test], 8)
    
    ######## real vs. predicted values
    #1) train set
    plot_real_predicted([y_train.values], [y_train_pred.values], '(' + str(loc) + ')', 'trainset', ['train set'], 
                        [col_train], min(y_train), max(y_train))
    # 2) control set
    line_min = int(min(min(y_control), min(y_test)))
    line_max = int(max(max(y_control), max(y_test)))

    
    plot_real_predicted([y_control.values, y_test], [y_control_pred.values, y_test_pred],
                        '(' + str(loc) + ')', 'control vs test', ['control set', 'test set'], 
                        [col_control, col_test], line_min, line_max)
    
    
    print('residuals control-mean: ', np.mean(resid_control), 'sd: ', np.std(resid_control))
    print('residuals test-mean: ', np.mean(resid_test), 'sd: ', np.std(resid_test))
    print('residuals training-mean: ', np.mean(resid_train), 'sd: ', np.std(resid_train))
    var_diff2 = np.var(resid_test) + np.var(resid_train)
    var_diff = np.var(resid_control) + np.var(resid_test)
    print('difference (control-test) residuals -mean: ', np.mean(resid_control) - np.mean(resid_test), 
          'sd: ', np.sqrt(var_diff))
    print('difference (test-train) residuals -mean: ', np.mean(resid_test)-np.mean(resid_train), 
          'sd: ', np.sqrt(var_diff2))
    var_diff3 = np.var(resid_control) + np.var(resid_train)
    print('difference (control-train) residuals -mean: ', np.mean(resid_control)-np.mean(resid_train), 
          'sd: ', np.sqrt(var_diff3))
    
    print('----------------------------------------------------------------')
    print('control and train')
    print('statistic:',ttest_ind(resid_train, resid_control, equal_var=False).statistic)
    print('pval:', ttest_ind(resid_train, resid_control, equal_var=False).pvalue)
    welch_dof(resid_train, resid_control)
    
    print('test and train')
    print('statistic:',ttest_ind(resid_train, resid_test, equal_var=False).statistic)
    print('pval:', ttest_ind(resid_test, resid_train, equal_var=False).pvalue)
    welch_dof(resid_test, resid_train)

    
    print('control and test')
    print('statistic:',ttest_ind(resid_train, resid_test, equal_var=False).statistic)
    print('pval:', ttest_ind(resid_control, resid_test, equal_var=False).pvalue)
    welch_dof(resid_control, resid_test)
    
else:
# model and plot for each station individually
    for loc in ["Tunnel","Steinlach","Hirschau"]:
        
        # fit the model (01.01.2018- 31.12.2021)
        y_train = df_train[loc]
        model = LinearRegression(fit_intercept=False)
        _ = model.fit(X, y_train)
        y_train_pred = pd.Series(model.predict(X), index=y_train.index)

        # make model prediction for 01.01.2022 - 27.11.2022
        X_test = dp.out_of_sample(len(cyclists22_after[loc])+ len(cyclists22_before[loc]))
        y_pred = pd.Series(model.predict(X_test), index=X_test.index)

        # get the control data (01.01.2022-05.04.2022)
        y_control = cyclists22_before[loc]
        y_control_pred = y_pred.filter(pd.date_range("2022"+start_date_before, "2022"+end_date_before, freq='W'))

        # get the test data (06.04.2022 -  27.11.2022)
        y_test = cyclists22_after[loc]
        y_test_pred = y_pred.filter(pd.date_range("2022"+start_date_after, "2022"+end_date_after, freq='W'))
        
        
        ######## plot the regression plot
        fig, a = plt.subplots(1,1)
        tit=f" {loc} Traffic - Seasonal Forecast"
        cyclists22_before[loc].plot(color=col_control,style='^', label="Test: Cycl. before blocking")
        cyclists22_after[loc].plot(color=col_test,style='*', label="Test: Cycl. after blocking")
        a = y_train.plot(color=col_train, style='.', label="Train data", title=tit)
        a = y_train_pred.plot(ax=a, label="Train pred.")
        a = y_test_pred.plot(ax=a, label="Test pred.", color=col_test)
        a = y_control_pred.plot(ax=a, label="Control pred.", color=col_control)
        _ = a.legend(loc="lower left")
        plt.show()

        ######## plot residuals in histogram
        # 1) train data
        resid_train = y_train.values - y_train_pred.values
        plot_residuals([resid_train], '(' + str(loc) + ')', 'trainset', ['train set'], [col_train], 50)

        # 2) control vs test
        resid_control = y_control.values - y_control_pred.values
        resid_test = y_test.values - y_test_pred.values
        plot_residuals([resid_test, resid_control], '(' + str(loc) + ')', 'control vs test', 
                       ['test set', 'control set'], [col_test, col_control], 8)

        ######## real vs. predicted values
        #1) train set
        plot_real_predicted([y_train.values], [y_train_pred.values], '(' + str(loc) + ')', 'trainset', ['train set'], 
                            [col_train], min(y_train), max(y_train))
        # 2) control set
        line_min = int(min(min(y_control), min(y_test)))
        line_max = int(max(max(y_control), max(y_test)))
        plot_real_predicted([y_test, y_control], [y_test_pred, y_control_pred],
                            '(' + str(loc) + ')', 'control vs test', ['test set', 'control set'], 
                            [col_test, col_control], line_min, line_max)

