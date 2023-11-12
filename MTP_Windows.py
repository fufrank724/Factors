import multiprocessing as mp
from util.backtest import *
from util.factor_util import *
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from tqdm import tqdm
import warnings
from time import sleep


warnings.filterwarnings("ignore")


INSAMPLE_END_DATE = datetime(2022,4,1)
VALID_END_DATE = datetime(2022,9,1)


df = pd.read_csv('E:/Python/Crypto/Factors/data/data_1h.csv')
df['BuyerRatio'] = df['takerBuyQuoteVol']/df['quoteAssetVolume']
df['BuyerPerTrade'] = df['takerBuyQuoteVol']/df['numberOfTrades']
df['VolumePerTrade'] = df['quoteAssetVolume']/df['numberOfTrades']

df = df.drop(['Volume','numberOfTrades'],axis= 1)

input_data_list = ['High','Low','Close','takerBuyQuoteVol','BuyerRatio']
formula_list = []
with open('E:/Python/Crypto/Factors/data/formulas/all_formula_list_20230208.csv','r') as f:
    formula_list += f.read().split('\n')

with open('E:/Python/Crypto/Factors/data/data_list/one_step_data_list_20230208.csv','r') as f:
    input_data_list += f.read().split('\n')
# with open('E:/Python/Crypto/Factors/data/data_list/two_step_data_list_20230208.csv','r') as f:
#     input_data_list += f.read().split('\n')
input_data_list = [input_data for input_data in input_data_list if 'PerTrade' not in input_data]

# np.random.shuffle(formula_list)
np.random.shuffle(input_data_list)
input_data_list = input_data_list[:]
formula_list = formula_list[:]
    

# base_data = 'Close'

close= df.pivot(values = 'Close',index = 'openTime',columns = 'symbol').astype(float)
Volume= df.pivot(values = 'quoteAssetVolume',index = 'openTime',columns = 'symbol').astype(float)

close.index = pd.to_datetime(close.index,unit = 'ms') + timedelta(hours=8)
Volume.index = pd.to_datetime(Volume.index,unit = 'ms') + timedelta(hours=8)


BM = crawl_cmcIndex()['CryptoMarket']
BM = BM.loc[datetime(2021,1,1):close.index[-1]]
BM = BM.pct_change()


market_filter = close.fillna(0)>0
dt = datetime.today().date().strftime('%Y%m%d')


def task(num,df,data_list,formula_list):
    ## backtest
    freq = 'D'
    backtest_metrics = {}
    backtest_performance = {}

    ret = close.sort_index().resample(freq).last().bfill().pct_change().fillna(0)

    for data_name in data_list:
        input_data = df.copy()
        input_data[data_name] = calc_input_data(df,data_name)
        input_data = input_data.pivot(values = data_name,index = 'openTime',columns = 'symbol').astype(float)
        input_data.index = pd.to_datetime(input_data.index,unit = 'ms') + timedelta(hours=8)



        for current,formula in enumerate(formula_list):
            operators = formula.split('.')
        
            # print(f"{data_name}: {formula}")


            # continue
            input_data = df.copy()
            input_data[data_name] = calc_input_data(df,data_name)
            input_data = input_data.pivot(values = data_name,index = 'openTime',columns = 'symbol').astype(float)
            input_data.index = pd.to_datetime(input_data.index,unit = 'ms') + timedelta(hours=8)

            factor = input_data.copy()
            for operator in operators:
                if operator in FORMULAS_PARAM_FREE:
                    factor = calc_factors(factor,operator,0)
                else:        
                    factor = calc_factors(factor,operator,24*7)

            # continue
            factor = factor.sort_index().resample(freq).last()
            volume = Volume.sort_index().resample(freq).sum().rolling(7).sum().fillna(0)
            volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
            GLOBAL_FILTER = market_filter & (volume_filter>0.5)
            cond = GLOBAL_FILTER 

            rk = factor[cond].rank(axis = 1,pct = True,ascending = True,method = 'dense')
            rk = (rk*10).apply(lambda x: np.floor(x))
            rk[rk==10] -= 1

            long_signal = rk.copy()
            long_signal[:] = 0
            short_signal = long_signal.copy()

            long_signal[rk==9] = 1 
            short_signal[rk==0] = -1 


            for reverse in [False,True]:
                if reverse:
                    long_signal *= -1
                    short_signal *= -1

                name = formula + ('_Reverse' if reverse else '')

                long_result = fast_backtest(ret,long_signal)#.sum(axis = 1)
                short_result = fast_backtest(ret,short_signal)#.sum(axis = 1)

                result = ((long_result + short_result)/2).sum(axis = 1)
                long_result = long_result.sum(axis = 1)
                short_result = short_result.sum(axis = 1)



                # show_return(result.loc[datetime(2021,1,1):])
                
                all_sample_result = show_performance_metrics(result.loc[datetime(2021,1,1):],show = False)
                insample_result= show_performance_metrics(result.loc[datetime(2021,1,1):INSAMPLE_END_DATE],show = False)
                outsample_result = show_performance_metrics(result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)


                long_alpha = (long_result.cumsum()/10**4).iloc[-1] - BM.add(1).cumprod().iloc[-1]
                short_alpha = (short_result.cumsum()/10**4).iloc[-1] - (BM*-1).add(1).cumprod().iloc[-1]


                
                msg = f'{data_name}.{name},{all_sample_result["Sharpe"]:.6f},{all_sample_result["FitnessValue"]:.6f},{all_sample_result["Monthly_Return_Score"]:.6f},{insample_result["Sharpe"]:.6f},{outsample_result["Sharpe"]:.6f},{long_alpha:.8f},{short_alpha:.8f}'
                with open(f'E:/Python/Crypto/Factors/output/backtest_result/D/backtest_metrics_{dt}.csv','a') as f:
                    f.write(msg+'\n')
                # plt.close()




if __name__=='__main__':
    num_process = 14





    ## Prepare Metric File

    with open(f'E:/Python/Crypto/Factors/output/backtest_result/D/backtest_metrics_{dt}.csv','w') as f:
        f.write('Strategy,Sharpe,Fitnessvalue,Monthly_Return_Score,Insample Sharpe,Outsample Sharpe,Long_Alpha,Short_Alpha\n')


    ## Run
    ix = int(len(input_data_list[:])/num_process)
    process_list = []
    for i in range(num_process):
        process_list.append(mp.Process(target = task, args = (i,df,input_data_list[i*ix:(i+1)*ix],formula_list)))
        process_list[i].start()
    i += 1
    process_list.append(mp.Process(target = task, args = (i,df,input_data_list[i*ix:(i+1)*ix],formula_list)))
    process_list[i].start()


    for i in range(num_process):
        process_list[i].join()

    print('Done')
