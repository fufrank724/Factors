import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,timezone
import json
import binance
from binance.client import Client
import requests
import time
import glob
import io
import pymongo


class BinanceDataManager:
    def __init__(
        self, api_key:str, api_secret:str,mongo_connenction = "mongodb+srv://Frank:cc840724@cluster0.l8bmlpk.mongodb.net/?retryWrites=true&w=majority"#,file_dir:str,
    ):
        # self.dir = file_dir
        self.key = api_key
        self.secret = api_secret
        self.pos_dict = {}
        self.file_dict = {}
        self.mode = ''
        self.client = Client(api_key=self.key, api_secret=self.secret,testnet = False)
        # self.info = self.exchange_info()

        self.mongo_client = pymongo.MongoClient(mongo_connenction)
        self.db = self.mongo_client["Binance"]

        self.update_futures_symbol_table()      
        self.update_spot_symbol_table()         

    def update_spot_symbol_table(self):
        table_exits = 'Spot_Symbol_table' in self.db.list_collection_names()
        symbol_table = self.db.Spot_Symbol_table
        result = self.client.get_exchange_info()['symbols']
        for si in result:
            try:
                if ('USDT' in si['symbol'] or'BUSD' in si['symbol'] ) and si['status'] == 'TRADING':
                    # info.append(si)

                    #clean filter type
                    for filter in si['filters']:
                        filterType = str.lower(filter['filterType'])#.copy()
                        del filter['filterType']
                        si[filterType] = filter.copy()

                    del si['filters']

                    #to mongodb
                    if not table_exits:
                        symbol_table.insert_one(si)
                    else:
                        symbol_table.update_many(
                            filter = {'symbol':si['symbol']},
                            update={
                                '$set': si,
                            },
                            upsert=True,
                        )
            except Exception as e:
                print('update symbol table failed', e)
        print('Symbol table update completed')


    def update_futures_symbol_table(self):
        table_exits = 'Futures_Symbol_table' in self.db.list_collection_names()
        symbol_table = self.db.Futures_Symbol_table
        result = self.client.futures_exchange_info()['symbols']
        for si in result:
            try:
                if 'USDT' in si['symbol'] and si['contractType']=='PERPETUAL' and si['status'] == 'TRADING':
                    # info.append(si)

                    #clean filter type
                    for filter in si['filters']:
                        filterType = str.lower(filter['filterType'])#.copy()
                        del filter['filterType']
                        si[filterType] = filter.copy()

                    del si['filters']

                    #to mongodb
                    if not table_exits:
                        symbol_table.insert_one(si)
                    else:
                        symbol_table.update_many(
                            filter = {'symbol':si['symbol']},
                            update={
                                '$set': si,
                            },
                            upsert=True,
                        )
            except Exception as e:
                print('update symbol table failed', e)
        print('Symbol table update completed')


    def crawl_futures_data(self,symbol,start_time,freq='1h'):
        seconds = 3600*24
        perpetual_hour = self.db[f'Perpetual_{freq}']


        # start_time = (self.client.futures_klines(symbol = symbol,interval = freq,limit = 1500,startTime = int(datetime(2020,1,1,0,0).timestamp()*1000)))[0][0]#/1000
        # start = symbol_table.loc[symbol].iloc[0]
        symbol = str.lower(symbol)
        current = int(datetime.now().timestamp()/seconds)*seconds*1000

        msg = f'=== Start crawl {symbol} === \nstart_time = {start_time}'
        self.log(msg)

        while start_time < current:
            current = int(datetime.now().timestamp()/seconds)*seconds*1000

            ## get symbol start_time
            try:
                data = (self.client.futures_klines(symbol = symbol,interval = freq,limit = 1500,startTime = int(start_time)))#[0][0]/1000
            except Exception as e:
                msg = f'Crawl {symbol}'
                self.log(msg,e)
                

            if len(data)==0:
                break
            start_time = (data[-1][6]+1)

            data = pd.DataFrame(data, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
            data['symbol'] = symbol
            data['index'] = data['symbol'] +"_" + data['openTime'].astype(str)

            data = list(data.T.to_dict().values())
            # break
            try:
                perpetual_hour.insert_many(
                    data
                )
            except Exception as e:
                msg = f'{symbol} {start_time} insert'
                self.log(msg,e)

            # print(data.tail())
            time.sleep(0.3)
            # break
        

    def load_future_data(self,reset_table,freq = '1h'):
        symbol_list = list(self.db.Futures_Symbol_table.find({},{'symbol':1,'_id':0}))
        symbol_list = [symbol['symbol'] for symbol in symbol_list] + ['LUNAUSDT','ANCUSDT','SCUSDT','BTTUSDT']

        #crawl data
        for symbol in symbol_list[:]:
            symbol = str.lower(symbol)
            perpetual_hour = self.db[f'Perpetual_{freq}']
            start = int(datetime(2020,1,1).timestamp())
            #check if exist
            if f'Perpetual_{freq}' not in self.db.list_collection_names():
                # perpetual_hour.create_index([("symbol", 1), ("openTime" , 1)])
                perpetual_hour.create_index([("index", 1)],unique = True,dropDups = True)
            else:
                ##if reload
                if not reset_table:
                    ##search first time of symbol
                    query = {"symbol": { "$eq": symbol } }
                    result = list(perpetual_hour.find(query,{'closeTime':1,'symbol':1,'_id':0}).sort('closeTime',-1).limit(1))#[:5]
                    # print(result)
                    if len(result) == 0:
                        # print('result = 0')
                        start = (self.client.futures_klines(symbol = symbol,interval = freq,limit = 1500,startTime = int(datetime(2020,1,1,0,0).timestamp()*1000)))[0][0]#/1000
                    else:
                        start = result[0]['closeTime'] +1 
                else:
                    ##reset database
                    
                    self.db.drop_collection(f'Perpetual_{freq}')
                    perpetual_hour = self.db[f'Perpetual_{freq}']
                    perpetual_hour.create_index([("index", 1)],unique = True,dropDups = True)



            # print(start)
            self.crawl_futures_data(symbol,start,freq)

            # break
        msg = f'{symbol} crawl completed'
        self.log(msg)
           
    def crawl_spot_data(self,symbol,start_time,freq='1h'):
        seconds = 3600*24
        spot = self.db[f'Spot_{freq}']


        # start_time = (self.client.futures_klines(symbol = symbol,interval = freq,limit = 1500,startTime = int(datetime(2020,1,1,0,0).timestamp()*1000)))[0][0]#/1000
        # start = symbol_table.loc[symbol].iloc[0]
        current = int(datetime.now().timestamp()/seconds)*seconds*1000

        msg = f'=== Start crawl {symbol} === \nstart_time = {start_time}'
        self.log(msg)

        while start_time < current:
            current = int(datetime.now().timestamp()/seconds)*seconds*1000

            ## get symbol start_time
            try:
                data = (self.client.get_klines(symbol = symbol,interval = freq,limit = 1500,startTime = int(start_time)))#[0][0]/1000
            except Exception as e:
                msg = f'Crawl {symbol}'
                self.log(msg,e)
                return
                

            if len(data)==0:
                break
            start_time = (data[-1][6]+1)

            data = pd.DataFrame(data, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
            data['symbol'] = symbol
            data['index'] = data['symbol'] +"_" + data['openTime'].astype(str)

            data = list(data.T.to_dict().values())
            # break
            try:
                spot.insert_many(
                    data
                )
            except Exception as e:
                msg = f'{symbol} {start_time} insert'
                self.log(msg,e)

            # print(data.tail())
            time.sleep(0.3)

        

    def load_spot_data(self,reset_table,freq = '1h'):
        symbol_list = list(self.db.Spot_Symbol_table.find({},{'symbol':1,'_id':0}))
        symbol_list = [symbol['symbol'] for symbol in symbol_list] + ['LUNAUSDT','ANCUSDT','SCUSDT','BTTUSDT']

        futures = list(self.db.Futures_Symbol_table.find({},{'symbol':1,'_id':0}))
        futures = [symbol['symbol'] for symbol in futures] + ['LUNAUSDT','ANCUSDT','SCUSDT','BTTUSDT']
        symbol_list = list(set(symbol_list) & set(futures))

        #crawl data
        for symbol in symbol_list[:1]:
            # symbol = str.lower(symbol)
            # symbol = symbol.upper()
            spot = self.db[f'Spot_{freq}']
            start = int(datetime(2020,10,1).timestamp())
            #check if exist
            if f'Spot_{freq}' not in self.db.list_collection_names():
                # perpetual_hour.create_index([("symbol", 1), ("openTime" , 1)])
                spot.create_index([("index", 1)],unique = True,dropDups = True)
            else:
                ##if reload
                if not reset_table:
                    ##search first time of symbol
                    query = {"symbol": { "$eq": str.lower(symbol) } }
                    result = list(spot.find(query,{'closeTime':1,'symbol':1,'_id':0}).sort('closeTime',-1).limit(1))#[:5]
                    # print(result)
                    if len(result) == 0:
                        # print('result = 0')
                        start = (self.client.get_klines(symbol =symbol,interval = freq,limit = 1500,startTime = int(datetime(2020,1,1,0,0).timestamp()*1000)))[0][0]#/1000
                    else:
                        start = result[0]['closeTime'] +1 
                else:
                    ##reset database
                    
                    self.db.drop_collection(f'Spot_{freq}')
                    spot = self.db[f'Spot_{freq}']
                    spot.create_index([("index", 1)],unique = True,dropDups = True)



            # print(start)
            self.crawl_spot_data(symbol,start,freq)

            # break
        msg = f'{symbol} crawl completed'
        self.log(msg)
           

    def log(self,msg,exception = None):
        now = datetime.now()
        timestamp = int(np.floor(now.timestamp()/3600/8)*3600*8)

        with open(f'./output/log/BinanceDataManager_{timestamp}.txt','a') as f:
            now = datetime.now()
            if exception:
                print(f'{msg} failed')
                f.write(f'{now}:{msg} failed:{exception}\n')
            else:
                print(msg)
                f.write(f'{now}:{msg}\n')
        

