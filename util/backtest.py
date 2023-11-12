import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import json
import requests
import pandas as pd
import seaborn as sns
import numba as nb
import matplotlib

def fast_backtest(ret,signal,fee = 4):



    w = signal.copy()
    w[:] = 1
    w = w*signal
    w[w>0] = w[w>0].divide(w[w>0].sum(axis = 1),axis = 0)
    w[w<0] = w[w<0].divide(w[w<0].abs().sum(axis = 1),axis = 0)
    w = w.divide(w.abs().sum(axis = 1),axis = 0).fillna(0)

    position = w * (10**4)
    position = position.replace([np.inf,-np.inf],0)


    cost = (position - position.shift(1)).abs()
    cost *= (10**-4) * fee



    result = ret * position.shift(1) #* w.shift(1)
    result -= cost
    # result = result.sum(axis = 1)
    # result.name = 'Strategy'

    return result



def backtest(price,condition,ordertype,fee_rate = (10**-4)*4,init_cap = 10**4):
    fee = fee_rate
    cap = init_cap
    size = 0
    side = 0

    profit = np.zeros(price.shape)

    close = price.to_numpy()
    signal = condition.to_numpy()
    for i in range(1,signal.shape[0]):
        if side == 0:
            if signal[i]==1:
                side = ordertype
                size = cap/close[i]
                profit[i] -= size * close[i] * fee
                
        #if has position
        elif side !=0 :
            profit[i] = size * (close[i]-close[i-1])* side
            if signal[i] == -1:
                profit[i] -= size * close[i] * fee

                side = 0       
                size = 0
    return profit

def backtest_multi_symbol(price,condition,ordertype,fee_rate = (10**-4)*4,init_cap = 10**4):
    ## constant
    fee = fee_rate
    cap = init_cap

    ## vars
    close = price.to_numpy()
    signal = condition.to_numpy()

    size = np.zeros(close[0].shape)
    side = np.zeros(close[0].shape)
    profit = np.zeros(close.shape)

    for i in range(1,signal.shape[0]):
        ## calc profit 
        profit[i] += size * (close[i]-close[i-1]) * side

        ## exit if has position
        mask = (side == ordertype) & (signal[i] == -1)
        profit[i][mask] -= (size * close[i] * fee)[mask]
        size[mask] = 0.
        side[mask] = 0.


        if np.sum(signal[i]==1) == 0:
            continue

        ## entry if no position
        mask = (signal[i] == 1)
        size[mask] = cap/np.sum(signal[i]==1) / close[i][mask]
        side[mask] = ordertype
        profit[i][mask] -= (size * close[i] * fee)[mask]

    return profit

def crawl_cmcIndex():
    start = int(datetime(2020,1,1).timestamp())
    end = int(datetime.now().timestamp())
    url = f'https://api.coinmarketcap.com/data-api/v3/global-metrics/quotes/historical?format=chart&interval=1d&timeEnd={end}&timeStart={start}'
    response = requests.request("GET", url)
    cmc_index = json.loads(response.text)['data']['quotes']
    cmc_index = pd.DataFrame([x['quote'][0] for x in cmc_index])[['timestamp','totalMarketCap']]
    cmc_index['timestamp'] = cmc_index['timestamp'].replace(['T','Z'],' ',regex = True)
    cmc_index.columns = ['DateTime','CryptoMarket']
    cmc_index = cmc_index.set_index('DateTime')
    cmc_index.index = pd.to_datetime(cmc_index.index)#.tz_localize('Asia/Taipei')#.tz_convert('UTC')
    cmc_index.index -= timedelta(hours = 16)
    cmc_index = cmc_index.pct_change().fillna(0).add(1).cumprod() *10**4

    return cmc_index
def show_return(result,bm = None,bm_name = 'CryptoMarket',figsize= (14,10),show_bm = False):

    fig = plt.figure(figsize=(12,10),constrained_layout=False)
    gs = fig.add_gridspec(20, 20)

    ax = fig.add_subplot(gs[:10, :])
    ax = result.cumsum().plot(ax = ax,title = 'Strategy Performance')
    result.cumsum()[result.cumsum()==result.cumsum().cummax()].reset_index().rename({0:'Profit'},axis = 1).plot.scatter(ax = ax,x=result.index.name,y='Profit',color= 'limegreen')

    if show_bm:
        ax2 = ax.twinx()
        bm = crawl_cmcIndex().pct_change().loc[result.index[0]:result.index[-1]]
        bm.add(1).cumprod().plot(ax = ax2)

    
    
    # plt.axvline(x=SAMPLE_END_DATE,c='firebrick',alpha= 0.5,ls = '--')
    # plt.axvline(x=VALID_END_DATE,c='silver',alpha= 0.7,ls = '--')
    # plt.axvline(x=INSAMPLE_END_DATE,c='silver',alpha= 0.5,ls = '--')

    # ax = fig.add_subplot(gs[:10, 10:])

    performance_metrics = show_performance_metrics(result,show = False)
    content = ''
    content += 'Ann. Sharpe:%.2f\n'%(performance_metrics['Ann. Sharpe'])
    # content += 'Ann. Ret/Mdd:%.2f\n'%(performance_metrics['Ret/Mdd'])
    # content += 'Ann. Return(%%):%.2f\n'%(performance_metrics['Ann. Return(%)'])
    content += 'MDD(%%):%.2f\n'%(performance_metrics['MDD(%)'])

    ax.text(0.85,0., content,
            verticalalignment='baseline',
            fontsize = 'large',
            transform=ax.transAxes)
    ax.set_xlabel('')


    ax = fig.add_subplot(gs[12:15, :])
    DD = result.cumsum().cummax()-result.cumsum()
    DD /= 10**4
    DD.plot(ax = ax,title = 'DrawDown',color=  'crimson')
    ax.set_xlabel('')

    ax = fig.add_subplot(gs[17:, :])
    monthly = result.reindex(pd.date_range(datetime(result.index.year[0],1,1),datetime(2023,12,31))).resample('M').sum().loc[datetime(2021,1,1):]

    month_table = {}
    for year in monthly.index.year.unique().to_list():
        month_table[year] = monthly[monthly.index.year == year].to_list()
        while len(month_table[year]) <12:
            month_table[year].append(0)

    month_table = pd.DataFrame(month_table)
    month_table.index += 1
    month_table = month_table.T / 10**4 

    colors = ["lemonchiffon","lightsalmon"]
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap",colors) 

    sns.heatmap(month_table,fmt = '.2%', annot=True,cmap=cmap1,cbar=False,ax = ax,vmax = month_table.stack().quantile(0.95),vmin = month_table.stack().quantile(0.25))
    ax.set_title('Monthly Return')
    plt.yticks(rotation=0)


    


def show_performance_metrics(profit,position = None,show = True,init_cap = 10**4):
    cum_profit = profit.cumsum()
    cum_profit_w = cum_profit.resample('W').last()
    monthly_return = profit.resample('M').sum()
    std = profit.std() if profit.std()!= 0 else 1
    ath_w = (np.arange(cum_profit_w.shape[0]) + 1)*0.0001

    ann_return = (cum_profit.iloc[-1]/init_cap)
    ann_return = ann_return ** (1/(cum_profit.shape[0]/365))
    mdd = (cum_profit.cummax()-cum_profit).max()/init_cap

    cummax_score = ((cum_profit_w==cum_profit_w.cummax()) * ath_w).sum()/np.sum(ath_w) #(cum_profit_w==cum_profit_w.cummax()).sum()/cum_profit_w.shape[0]
    
    result = {
        'Ann. Return(%)':ann_return*100,
        'MDD(%)':(cum_profit.cummax()-cum_profit).max()/init_cap*100,
        'Ret/Mdd':ann_return/mdd,
        'Ann. Sharpe':profit.mean()/std* (365) ** 0.5,
        'ATH_Score':(cum_profit == cum_profit.cummax()).sum() / profit.shape[0],
        'Monthly_Return_Score':(monthly_return>0).sum()/monthly_return.shape[0],
        'Cummax_Score':cummax_score,
        'FitnessValue': profit.mean()/std* (profit.shape[0]) ** 0.5 * (cummax_score + (1 if cum_profit.iloc[-1]<0 else 0)),
        # 'win_rate':(profit>0).sum()/(profit!=0).sum(),
        # 'win_ratio':profit[profit>0].mean()/-profit[profit<0].mean(),

    }   
  



    if position:
        result['num_trades'] = (position!=position.shift(1)).sum().sum()
  
    if show:
        for key,value in result.items():
            print(f'{key}: {value:.2f}')
    return result    