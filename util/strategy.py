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
from sqlalchemy import false

FEE = 0.0004

class Backtest:
    def __init__(self,strategy):        
        pass

    def run(self):
        pass

class Agent:
    def __init__(self) -> None:
        self.positions = []
        self.price = []



    def close_all_position(self):
        profit = self.positions[0]
        profit[:] = 0
        
        for position in self.positions:
            profit += position.size * (self.price[-1] - self.price[-2])
        
        self.positions.clear()
        return profit
        

    def open_position(self,size,sl = None,tp = None):
        self.positions.append(Position(size,sl,tp))
        

    

class Position:
    def __init__(self,size,sl = None,tp = None) -> None:
        self.size = 0
        self.sl = None
        self.tp = None


class Strategy:
    def __init__(self) -> None:
        self.symbol = pd.Series()

        pass
    def load_data(self):
        pass

    def act(self,price):

        long_cond = false
        short_cond = false

        if side!=0:
            pass


        if side == 0:
            # if self.signal.iloc[i]==1:
            #     price.iloc[i] = price.iloc[i] * (1+cost)
            #     size = init_cap/price.iloc[i]
            #     side = 1
            if self.signal.iloc[i]==-1:
                side = -1
                size = init_cap/price.iloc[i]
                profit[i] -= size * price.iloc[i] * FEE
                record[i] = side#
                record_fee[i] = size * price.iloc[i] * FEE
                
        #if has position
        elif side !=0 :
            profit[i] = size * (price.iloc[i]-price.iloc[i-1])* side
            if self.signal.iloc[i] == side*-1:
                profit[i] = size * (price.iloc[i]-price.iloc[i-1])* side
                profit[i] -= size * price.iloc[i] * FEE
                record[i] = side
                record_fee[i] = size * price.iloc[i] * FEE

                side = 0       
                size = 0

