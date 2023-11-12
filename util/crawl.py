from locale import currency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,timezone
import json
from binance.client import Client
import requests
import pymongo
import time

## Binance

def crawl_binance_futures(symbol,start,end=None,USDT=True,freq = '1m'):

    if not end:
        end = datetime.now().timestamp()*1000

    # usd = 'USDT' if USDT else 'BUSD'

    currency = symbol #+ usd

    key = 'U2R4Ixr3153LPd4yG6fjwQSBFHE25S1Pei2u5tLfht1wC5NlHUs86kjEGIF8IWE9'
    secret = 'sYBOO9tESV4QcQZOncUWTO4cIicoCSvpkKdCAXeAp5qYP2rK2WEg1Fzcu04cqTYB'
    client = Client(api_key=key, api_secret=secret,testnet = False)
    seconds = 3600*24

    start_time = start

    ls = []

    while start_time < end:

        data = (client.futures_klines(symbol = currency,interval = freq,limit = 1500,startTime = int(start_time)))#[0][0]/1000
        if len(data)==0:
            break
        start_time = (data[-1][6]+1)

        data = pd.DataFrame(data, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
        ls.append(data)
        time.sleep(0.6)
    return pd.concat(ls).set_index('openTime')

def crawl_binance_spot(symbol,start,end = None,USDT=True,freq = '1m'):
    
    if not end:
        end = datetime.now().timestamp()*1000
    
    usd = 'USDT' if USDT else 'BUSD'

    currency = symbol + usd

    key = 'U2R4Ixr3153LPd4yG6fjwQSBFHE25S1Pei2u5tLfht1wC5NlHUs86kjEGIF8IWE9'
    secret = 'sYBOO9tESV4QcQZOncUWTO4cIicoCSvpkKdCAXeAp5qYP2rK2WEg1Fzcu04cqTYB'
    client = Client(api_key=key, api_secret=secret,testnet = False)
    seconds = 3600*24

    start_time = start

    ls = []

    while start_time < end:

        data = (client.get_klines(symbol = currency,interval = freq,limit = 1500,startTime = int(start_time)))#[0][0]/1000
        if len(data)==0:
            break
        start_time = (data[-1][6]+1)

        data = pd.DataFrame(data, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
        ls.append(data)
        time.sleep(0.2)
    return pd.concat(ls).set_index('openTime')
## MEXC
## futures data only provide recent 30 days
## index:time = openTime
def crawl_mexc_futures(symbol,start,end = None):
    currency = symbol+"_USDT"
    ls = []

    end = 0
    if not end:
        end = datetime.now().timestamp()*1000


    start = int(start/1000)
    end = int(end/1000)

    while start < end:
        url = f'https://contract.mexc.com/api/v1/contract/kline/{currency}?interval=Min1&start={start}'#&start_time={start}&limit=100'
        r = requests.get(url)
        data = pd.DataFrame(r.json()['data'])
        if data.shape[0] == 0:
            break
        
        ls.append(data)
        start = r.json()['data']['time'][-1] + 60
    
    data = pd.concat(ls)
    data = data.rename({'time':'closeTime'},axis = 1).set_index('closeTime').astype(float)
    data.index *= 1000
    return data

def crawl_mexc_spot(symbol,start,end = None):
    currency = symbol+"_USDT"
    ls = []

    end = 0
    if not end:
        end = datetime.now().timestamp()*1000

    start = int(start/1000)
    end = int(end/1000)

    while start < end:

        url = f'https://www.mexc.com/open/api/v2/market/kline?symbol={currency}&interval=1m&limit=1000&start_time={start}'#&start_time={start}&limit=100'
        r = requests.get(url)
        data = pd.DataFrame(r.json()['data'])
        if data.shape[0] == 0:
            break
        
        ls.append(data)

        start = r.json()['data'][-1][0] + 60
    
    data = pd.concat(ls)
    data.columns = ['time','open','close','high','low','volume','amount']
    data = data.rename({'time':'closeTime'},axis = 1).set_index('closeTime').astype(float)
    data.index *= 1000
    return data    

## FTX
def crawl_ftx_spot(symbol,start,end,USDT = False):
    ls = []
    currency = symbol
    if USDT:
        currency = symbol + '/USDT'
    else:
        currency = symbol + '/USD'
    
    resolution = 60
    start = int(start/1000)
    while start < end/1000:
        limit = start + 60*1000
        # print(start/1000,limit)
        url = f'https://ftx.com/api/markets/{currency}/candles?resolution={resolution}&start_time={start}&end_time={limit}'
        r = requests.get(url)
        # print(r.json())
        data = pd.DataFrame(r.json()['result'])
        # break
        
        if data.shape[0] == 0:
            break
        
        ls.append(data)

        start = int(r.json()['result'][-1]['time']/1000) + 60  
        
    data = pd.concat(ls)
    data = data.set_index('time').drop('startTime',axis = 1).astype(float)
    return data    

def crawl_ftx_futures(symbol,start,end):
    ls = []
    currency = symbol + '-PERP'
        
    resolution = 60
    start = int(start/1000)
    while start < end/1000:
        limit = start + 60*1000
        # print(start/1000,limit)
        url = f'https://ftx.com/api/markets/{currency}/candles?resolution={resolution}&start_time={start}&end_time={limit}'
        r = requests.get(url)
        # print(r.json())
        data = pd.DataFrame(r.json()['result'])
        # break
        
        if data.shape[0] == 0:
            break
        
        ls.append(data)

        start = int(r.json()['result'][-1]['time']/1000) + 60  
        
    data = pd.concat(ls)
    data = data.set_index('time').drop('startTime',axis = 1).astype(float)
    return data   

## Bybit
### bybit spot only provide 3500 candles
def crawl_bybit_spot(symbol):
    currency = symbol+"USDT"
    ls = []

    url = f'https://api.bybit.com/spot/quote/v1/kline?symbol={currency}&interval=1m'

    r = requests.get(url)
    start = int(r.json()['result'][-1][0]) - 4000*60*1000 
    end = r.json()['result'][-1][0]
    limit = start + 1000*60*1000

    while start < end:

        url = f'https://api.bybit.com/spot/quote/v1/kline?symbol={currency}&interval=1m&limit=1000&startTime={start}&endTime={limit}'
        r = requests.get(url)
        data = pd.DataFrame(r.json()['result'])
        data.columns = ['startTime','open','high','low','close','volume','endTime','quateAssetVolume','trades','takersBaseVolume','takerQuoteVolume']
        data = data.set_index('startTime')
        
        if data.shape[0] == 0:
            print('no data')
            break
        
        ls.append(data)

        start = r.json()['result'][-1][0] + 60*1000
        limit = start + 1000*60*1000
        print(start,end)
    

    return pd.concat(ls)    


def crawl_okx_fut(symbol,start,end):
    currency = symbol+"USDT"
    ls = []

    start_time = start

    # Replace with your OKEx API key and secret
    api_key = '1362d152-9398-45e7-a876-1d8e39a739c2'
    api_secret = 'YOUR_API_SECRET'



    url = f'https://www.okex.com/api/v5/market/history-candles'
    symbol = f'{symbol}-usdt-swap'.upper()
    interval = '1m'

    

    start_time = start
    end_time = start_time + 60*1000 * 100


    while start_time < end:
        
        end_time = start_time + 60*1000 * 100
        
        params = {
                'instId': symbol,
                'bar': interval,
                'before': start_time,
                'after': end_time,
                'limit': 1000
            }        

        response = requests.get(url, params=params, headers={'Authorization': f'Bearer {api_key}'})
        data = pd.DataFrame(response.json()['data'])
        data.columns = ['openTime','Open','High','Low','Close','vol','volCcy','volCcyQuote','confirm']
        # data = data.set_index('startTime')
        
        if data.shape[0] == 0:
            print('no data')
            break
        
        ls.append(data)

        start_time = int(response.json()['data'][0][0])
        # print(response.json()['data'][0][0],response.json()['data'][-1][0])
        # print(start_time)
        
        time.sleep(0.2)
    

    return pd.concat(ls)    
