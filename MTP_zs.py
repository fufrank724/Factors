import multiprocessing as mp
from util.backtest import *
from util.factor_util import *
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from tqdm import tqdm
import warnings
from time import sleep
import statsmodels.api as sm
import os
from numba import njit

warnings.filterwarnings("ignore")



def task(strategy_list):
    ## backtest

    data_name = ''
    formula = ''
    for strategy in strategy_list:
        print(strategy)
        reverse = '_Reverse' in strategy
        data = strategy.replace('_Reverse','').split('.')
        
        
        if data[0] != data:
            data_name = data[0]
            input_data = calc_input_data(df_data,data_name)
        
        if formula != '.'.join(data[1:]):
            formula = '.'.join(data[1:])
            factor = calc_factors(input_data,formula,params=[24,24,24])
            factor = factor.sort_index()





        # continue
        
        cond = GLOBAL_FILTER 


        zscore = factor.add(-factor.mean(axis = 1),axis = 0).div(factor.std(axis = 1),axis = 0)

        signal = zscore.copy()
        signal[:] = 0
        signal[(zscore>4) & cond] = -1
        signal[(zscore<-4) & cond] = 1

        size = signal.copy()
        size = 1/df_data['Close'].div((10**4/signal.abs().sum(axis = 1)),axis = 0).replace([np.inf,-np.inf],0)
        size = size.ffill().fillna(0).replace([np.inf,-np.inf],0) * signal


        FEE = 0

        cost = size.diff().abs() * df_data['Close'] * FEE *10**-4

        result = size.shift(1) * df_data['Close'].diff() - cost
        long_result = size[(size>0)].fillna(0).shift(1) * df_data['Close'].diff() - size[(size>0)].fillna(0).shift(1).diff().abs() * df_data['Close'] * FEE *10**-4
        short_result = size[(size<0)].fillna(0).shift(1) * df_data['Close'].diff() - size[(size<0)].fillna(0).shift(1).diff().abs() * df_data['Close'] * FEE *10**-4

        result = result.sum(axis = 1).resample('D').sum()
        long_result = long_result.sum(axis = 1).resample('D').sum()
        short_result = short_result.sum(axis = 1).resample('D').sum()
        

        # show_return(result.loc[datetime(2021,1,1):VALID_END_DATE],figsize= (12,4))


        ## Performance metrics
        all_sample_result = show_performance_metrics(result.loc[datetime(2021,1,1):SAMPLE_END_DATE],show = False)
        insample_result= show_performance_metrics(result.loc[datetime(2021,1,1):INSAMPLE_END_DATE],show = False)
        outsample_result = show_performance_metrics(result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)

        long_all_sample_result = show_performance_metrics(long_result.loc[datetime(2021,1,1):SAMPLE_END_DATE],show = False)
        short_all_sample_result = show_performance_metrics(short_result.loc[datetime(2021,1,1):SAMPLE_END_DATE],show = False)

        long_insample_result= show_performance_metrics(long_result.loc[datetime(2021,1,1):INSAMPLE_END_DATE],show = False)
        long_outsample_result = show_performance_metrics(long_result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)
        short_insample_result= show_performance_metrics(short_result.loc[datetime(2021,1,1):INSAMPLE_END_DATE],show = False)
        short_outsample_result = show_performance_metrics(short_result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)


        msg = f'{data_name}.{formula},{all_sample_result["Ann. Sharpe"]:.6f},{all_sample_result["Ann. Sharpe"]:.6f},{all_sample_result["FitnessValue"]:.6f},{insample_result["Ann. Sharpe"]:.6f},{outsample_result["Ann. Sharpe"]:.6f},{long_all_sample_result["FitnessValue"]:.6f},{short_all_sample_result["FitnessValue"]:.6f},{long_insample_result["FitnessValue"]:.6f},{short_insample_result["FitnessValue"]:.6f},{long_outsample_result["FitnessValue"]:.6f},{short_outsample_result["FitnessValue"]:.6f}'

        with open(FNAME,'a') as f:
            f.write(msg+'\n')





if __name__=='__main__':
    num_process = 30

    INSAMPLE_END_DATE = datetime(2021,12,31)
    VALID_END_DATE = datetime(2022,6,1)
    SAMPLE_END_DATE = datetime(2023,12,31)

    FREQ = 'D'
    FNAME = f'/home/frank/document/Python/Factors/output/backtest_result/ZS/level1_metrics.csv'


    df = pd.read_csv('/home/frank/document/Python/Factors/data/data_1h.csv')
    df['BuyerRatio'] = df['takerBuyQuoteVol']/df['quoteAssetVolume']
    df['BuyerPerTrade'] = df['takerBuyQuoteVol']/df['numberOfTrades']
    df['VolumePerTrade'] = df['quoteAssetVolume']/df['numberOfTrades']

    df = df.drop(['Volume','numberOfTrades'],axis= 1)
    print('Dataframe prepared completed')

    ## Pivot tables
    col_list = ['Open','High','Low','Close','quoteAssetVolume','takerBuyQuoteVol','BuyerRatio']
    df_data = {}
    for col in col_list:
        df_data[col] = df.pivot(values = col,index = 'openTime',columns = 'symbol').astype(float)
        df_data[col].index = pd.to_datetime(df_data[col].index,unit = 'ms') #+ timedelta(hours=8)

    market_filter = ~((df_data['quoteAssetVolume'] == 0) | df_data['quoteAssetVolume'].isna())
    volume = df_data['quoteAssetVolume'].sort_index().rolling(72).sum().fillna(0)
    volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
    GLOBAL_FILTER = market_filter & (volume_filter>0.5)
    # GLOBAL_FILTER = GLOBAL_FILTER.resample(FREQ).last()
    cond = GLOBAL_FILTER

    ret = df_data['Close'].sort_index().pct_change().fillna(0)
    print("Pivot table completed")


    # Prepare Metric File
    prev_result = []
    dt = datetime.today().date().strftime('%Y%m%d')
    if os.path.exists(FNAME):
        print('file exits')
        prev_result = pd.read_csv(FNAME).Strategy.to_list()

    else:        
        with open(FNAME,'w') as f:
            f.write('Strategy,Sharpe,Fitnessvalue,Insample Sharpe,Outsample Sharpe,Long Fitness,Short Fitness,Long Insample Fitness,Short Insample Fitness,Long Outsample Fitness,Short Outsample Fitness\n')



    ## Calc Data
    formula_list = []
    # input_data_list = ['High','Low','Close','takerBuyQuoteVol','BuyerRatio']
    input_data_list = []

    with open('/home/frank/document/Python/Factors/data/formulas/all_formula_list_20230221.csv','r') as f:
        formula_list += f.read().split('\n')

    with open('/home/frank/document/Python/Factors/data/data_list/all_data_list_20230221.csv','r') as f:
        input_data_list += f.read().split('\n')

    # np.random.shuffle(formula_list)

    input_data_list = [input_data for input_data in input_data_list if 'PerTrade' not in input_data]
    combinations = list(itertools.product(input_data_list,formula_list,['','Reverse']))
    combinations = ['.'.join(combination[:-1]) + (('_'+combination[-1]) if combination[-1] != '' else '') for combination in combinations]
    combinations = list(set(combinations) - set(prev_result))
    combinations = combinations[:30]
    print("Generate Combinations completed")


    
    BM = crawl_cmcIndex()['CryptoMarket']
    BM = BM.loc[datetime(2021,1,1):df_data['Close'].index[-1]]
    BM = BM.pct_change()



    # Run
    process_list = []
    ix = int(len(combinations[:])/num_process)
    for i in range(num_process):
        process_list.append(mp.Process(target = task, args = (combinations[i*ix:(i+1)*ix],)))
        process_list[i].start()
    i += 1
    process_list.append(mp.Process(target = task, args = (combinations[i*ix:],)))
    process_list[i].start()


    for i in range(len(process_list)):
        process_list[i].join()

    print('Done')
