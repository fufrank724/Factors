import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import json
import requests
import pandas as pd

def fast_backtest(ret,signal,fee = 8):



    w = signal.copy()
    w[:] = 1
    w = w*signal
    w[w>0] = w[w>0].divide(w[w>0].sum(axis = 1),axis = 0)
    w[w<0] = w[w<0].divide(w[w<0].abs().sum(axis = 1),axis = 0)
    w = w.divide(w.abs().sum(axis = 1),axis = 0)

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

def show_return(result,bm = None,bm_name = 'CryptoMarket'):
    if not isinstance(bm, pd.Series) :
        cmc_index = crawl_cmcIndex()
        bm = cmc_index
    
    fig = plt.figure(figsize=(14, 10),constrained_layout=False)
    gs = fig.add_gridspec(20, 20)

    ax = fig.add_subplot(gs[:8, :])
    ax = result.cumsum().plot(ax = ax,title = 'Equity')
    ax2 = ax.twinx()
    bm.loc[result.index[0]:result.index[-1]].plot(ax = ax2,color = 'green',grid = False,alpha = 0.3)

    # ax.legend( ['Strategy',bm_name],loc="upper left")
    ax2.legend( [bm_name],loc="upper left")
    # ax.legend([ax, ax2],['Strategy',bm_name])


    # ax = fig.add_subplot(gs[11:, :10])
    # (position>0).sum(axis= 1).plot(ax = ax,title = 'Num of holding currencies')
    # (position<0).sum(axis= 1).plot(ax = ax)
    # ax.legend(['Long','Short'])


    ax = fig.add_subplot(gs[11:, :])
    monthly = result.resample('M').sum().loc[datetime(2021,1,1):]
    monthly.index = monthly.index.strftime('%Y/%m')
    monthly_g = monthly.copy()
    monthly_r = monthly.copy()
    monthly_g[monthly_g<0] = 0
    monthly_r[monthly_r>0] = 0

    monthly_g.plot.bar(ax = ax,title = 'Monthly Return')
    monthly_r.plot.bar(ax = ax,color = 'red')
  

  
def show_performance_metrics(profit,position = None,show = True,init_cap = 10**4):
    std = profit.std() if profit.std()!= 0 else 1
    result = {
    'net_profit(%)':profit.cumsum().iloc[-1]/init_cap*100,
    'MDD(%)':(profit.cumsum().cummax()-profit.cumsum()).max()/init_cap*100,
    'profit_mdd':profit.cumsum().iloc[-1]/(profit.cumsum().cummax()-profit.cumsum()).max(),
    'sharpe':profit.mean()/std* (profit.shape[0]) ** 0.5,
    'win_rate':(profit>0).sum()/(profit!=0).sum(),
    'win_ratio':profit[profit>0].mean()/-profit[profit<0].mean(),
    }         



    if position:
        result['num_trades'] = (position!=position.shift(1)).sum().sum()
  
    if show:
        for key,value in result.items():
            print(f'{key}: {value:.2f}')
    return result    