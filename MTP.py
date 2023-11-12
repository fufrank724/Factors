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

@njit
def check_combinations(combinations,prev_result):
    return ['.'.join(combination[:-1]) + (('_'+combination[-1]) if combination[-1] != '' else '') for combination in combinations[:] if '.'.join(combination[:-1]) + (('_'+combination[-1]) if combination[-1] != '' else '') not in prev_result]

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
            factor = calc_factors(input_data,formula)
            factor = factor.sort_index().resample(FREQ).last()



        if reverse:
            factor *= -1

        # continue
        
        cond = GLOBAL_FILTER 
        selected = 10
        select = cond.sum(axis = 1) * selected * 0.01
        select = select.apply(lambda x:max(np.floor(x),3))

        rk = factor[cond].rank(axis = 1,ascending = True,method = 'dense')

        long_signal = rk.copy()
        long_signal[:] = 0
        short_signal = long_signal.copy()

        long_signal[rk.gt(cond.sum(axis = 1) - select,axis = 0)] = 1 
        short_signal[rk.le(select,axis = 0)] = -1 

        # rk = factor[cond].rank(axis = 1,pct = True,method = 'dense')
        # rk = rk.add(-rk.mean(axis = 1),axis= 0)
        # long_signal = rk.copy()
        # short_signal = rk.copy()
        # long_signal[rk<0] = 0
        # short_signal[rk>0] = 0


        name = formula + ('_Reverse' if reverse else '')

        long_result = fast_backtest(ret,long_signal,fee = 0)#.sum(axis = 1)
        short_result = fast_backtest(ret,short_signal,fee = 0)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)

        # signal = long_signal + short_signal
        # turnover = signal.fillna(0).diff().fillna(0).abs().sum(axis = 1)/market_filter.sum(axis = 1).resample('D').last() * 100
        
        long_signal = long_signal.div(long_signal.sum(axis = 1),axis = 0)
        short_signal = short_signal.div(short_signal.sum(axis = 1),axis = 0) * -1

        signal = (long_signal + short_signal)/2
        turnover = signal.diff().abs().sum(axis = 1)        
        turnover = turnover.loc[datetime(2020,1,1):SAMPLE_END_DATE]
        IS_turnover = turnover.loc[datetime(2020,1,1):INSAMPLE_END_DATE].mean()
        OS_turnover = turnover.loc[INSAMPLE_END_DATE:VALID_END_DATE].mean()

    

        # show_return(result.loc[datetime(2020,1,1):VALID_END_DATE],figsize= (12,4))


        ## Performance metrics
        all_sample_result = show_performance_metrics(result.loc[datetime(2020,1,1):SAMPLE_END_DATE],show = False)
        insample_result= show_performance_metrics(result.loc[datetime(2020,1,1):INSAMPLE_END_DATE],show = False)
        outsample_result = show_performance_metrics(result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)

        long_all_sample_result = show_performance_metrics(long_result.loc[datetime(2020,1,1):SAMPLE_END_DATE],show = False)
        short_all_sample_result = show_performance_metrics(short_result.loc[datetime(2020,1,1):SAMPLE_END_DATE],show = False)

        long_insample_result= show_performance_metrics(long_result.loc[datetime(2020,1,1):INSAMPLE_END_DATE],show = False)
        long_outsample_result = show_performance_metrics(long_result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)
        short_insample_result= show_performance_metrics(short_result.loc[datetime(2020,1,1):INSAMPLE_END_DATE],show = False)
        short_outsample_result = show_performance_metrics(short_result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)

        # if outsample_result['Sharpe']>insample_result['Sharpe']*0.75 and all_sample_result['Sharpe']>1 and outsample_result['Sharpe']>1 and all_sample_result['Sharpe'] != all_sample_result['FitnessValue']:
        # pd.DataFrame({
        #     'Long':long_result,
        #     'Short':short_result
        # }).to_csv(f'/home/frank/document/Python/Factors/output/nav/level1/{input_data_list.index(data_name)}_{formula_list.index(name)}.csv')

        ## IC
        ret_t1 = ret[market_filter].shift(-1).loc[datetime(2020,1,1):].stack().reset_index().rename({0:'return(t+1)'},axis = 1)
        ff = factor.resample('D').last().loc[datetime(2020,1,1):]
        ff = ff.stack().reset_index().rename({0:'factor'},axis = 1)
        df = pd.merge(ff,ret_t1,on = ['openTime','symbol'])
        # df['Month'] = df.openTime.dt.month + df.openTime.dt.year*100
        IC = df.groupby('openTime').apply(lambda x: x.corr().iloc[1,0])
        IC = IC.loc[datetime(2020,1,1):SAMPLE_END_DATE]
        IR = IC.mean()/IC.std()#.plot.bar()**

        IS_IC = IC.loc[datetime(2020,1,1):INSAMPLE_END_DATE]
        IS_IR = IS_IC.quantile(0.5)/IS_IC.std()
        IS_IC = IS_IC.quantile(0.5)
        OS_IC = IC.loc[INSAMPLE_END_DATE:VALID_END_DATE]
        OS_IR = OS_IC.quantile(0.5)/OS_IC.std()
        OS_IC = OS_IC.quantile(0.5)
        IC = IC.quantile(0.5)

        ## RankIC
        rk = factor.rank(axis = 1,ascending = True,method = 'dense')
        ff = rk.resample('D').last().loc[datetime(2020,1,1):]#.stack().reset_index().rename({0:'factor'},axis = 1)
        ff = ff.stack().reset_index().rename({0:'factor'},axis = 1)
        df = pd.merge(ff,ret_t1,on = ['openTime','symbol'])
        Rank_IC = df.groupby('openTime').apply(lambda x: x.corr().iloc[1,0])
        Rank_IR = Rank_IC.quantile(0.5)/Rank_IC.std()#.plot.bar()

        IS_Rank_IC = Rank_IC.loc[datetime(2020,1,1):INSAMPLE_END_DATE]
        IS_Rank_IR = IS_Rank_IC.quantile(0.5)/IS_Rank_IC.std()
        IS_Rank_IC = IS_Rank_IC.quantile(0.5)
        OS_Rank_IC = Rank_IC.loc[INSAMPLE_END_DATE:VALID_END_DATE]
        OS_Rank_IR = OS_Rank_IC.quantile(0.5)/OS_Rank_IC.std()
        OS_Rank_IC = OS_Rank_IC.quantile(0.5)
        Rank_IC = Rank_IC.quantile(0.5)

        winloss_ratio = -result[result>0].quantile(0.5)/result[result<0].quantile(0.5)        
        IS_winloss_ratio = -result[result>0].loc[datetime(2020,1,1):INSAMPLE_END_DATE].quantile(0.5)/result[result<0].loc[datetime(2020,1,1):INSAMPLE_END_DATE].quantile(0.5)
        OS_winloss_ratio = -result[result>0].loc[INSAMPLE_END_DATE:VALID_END_DATE].quantile(0.5)/result[result<0].loc[INSAMPLE_END_DATE:VALID_END_DATE].quantile(0.5)


        ## 
        # msg = f'\n{input_data_list.index(data_name)}.{formula_list.index(name)},{all_sample_result["Ann. Sharpe"]:.6f},{all_sample_result["FitnessValue"]:.6f},{turnover.mean():.6f},{IC:.6f},{IR:.6f},{Rank_IC:.6f},{Rank_IR:.6f},{insample_result["Ann. Sharpe"]:.6f},{outsample_result["Ann. Sharpe"]:.6f},{long_all_sample_result["FitnessValue"]:.6f},{short_all_sample_result["FitnessValue"]:.6f},{long_insample_result["FitnessValue"]:.6f},{short_insample_result["FitnessValue"]:.6f},{long_outsample_result["FitnessValue"]:.6f},{short_outsample_result["FitnessValue"]:.6f},{IS_turnover:.6f},{OS_turnover:.6f},{IS_IC:.6f},{IS_IR:.6f},{OS_IC:.6f},{OS_IR:.6f},{IS_Rank_IC:.6f},{IS_Rank_IR:.6f},{OS_Rank_IC:.6f},{OS_Rank_IR:.6f}'
        msg = f'\n{data_name}.{name},{all_sample_result["Ann. Sharpe"]:.6f},{all_sample_result["FitnessValue"]:.6f},{turnover.mean():.6f},{winloss_ratio},{IC:.6f},{IR:.6f},{Rank_IC:.6f},{Rank_IR:.6f},{insample_result["Ann. Sharpe"]:.6f},{outsample_result["Ann. Sharpe"]:.6f},{long_all_sample_result["FitnessValue"]:.6f},{short_all_sample_result["FitnessValue"]:.6f},{IS_winloss_ratio},{OS_winloss_ratio},{long_insample_result["FitnessValue"]:.6f},{short_insample_result["FitnessValue"]:.6f},{long_outsample_result["FitnessValue"]:.6f},{short_outsample_result["FitnessValue"]:.6f},{IS_turnover:.6f},{OS_turnover:.6f},{IS_IC:.6f},{IS_IR:.6f},{OS_IC:.6f},{OS_IR:.6f},{IS_Rank_IC:.6f},{IS_Rank_IR:.6f},{OS_Rank_IC:.6f},{OS_Rank_IR:.6f}'

        with open(FNAME,'a') as f:
            f.write(msg)
        # return





if __name__=='__main__':
    num_process = 30

    INSAMPLE_END_DATE = datetime(2021,12,31)
    VALID_END_DATE = datetime(2022,6,1)
    SAMPLE_END_DATE = datetime(2022,9,30)

    FREQ = 'H'
    FNAME = f'/home/frank/document/Python/Factors/output/1H/level1_IC_metrics2.csv'


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
    df_data['takerBuyQuoteVol'] = df_data['takerBuyQuoteVol'] * 2 - df_data['quoteAssetVolume']

    market_filter = ~((df_data['quoteAssetVolume'] == 0) | df_data['quoteAssetVolume'].isna())
    volume = df_data['quoteAssetVolume'].sort_index().rolling(7*24).sum().fillna(0)
    volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
    GLOBAL_FILTER = market_filter & (volume_filter>0.5)
    GLOBAL_FILTER = GLOBAL_FILTER.resample('D').last()

    ret = df_data['Close'].sort_index().resample(FREQ).last().bfill().pct_change().fillna(0)
    print("Pivot table completed")




    ## Calc Data
    formula_list = []
    # input_data_list = ['High','Low','Close','takerBuyQuoteVol','BuyerRatio']
    input_data_list = []

    with open('/home/frank/document/Python/Factors/data/formulas/all_formula_list_20230221.csv','r') as f:
        formula_list += f.read().split('\n')

    with open('/home/frank/document/Python/Factors/data/data_list/all_data_list_20230221.csv','r') as f:
        input_data_list += f.read().split('\n')

    np.random.shuffle(formula_list)

    # Prepare Metric File
    prev_result = []
    dt = datetime.today().date().strftime('%Y%m%d')
    if os.path.exists(FNAME):
        print('file exists')
        with open(FNAME,'r') as f:
            prev_result += f.read().split('\n')
            prev_result = [data.split(',')[0] for data in prev_result]
            # prev_result = [data.split(',')[0] for data in prev_result]
            # prev_result = [input_data_list[int(data.split('.')[0])] + '.' + formula_list[int(data.split('.')[1])] for data in prev_result[1:]]
    else:        
        with open(FNAME,'w') as f:
            f.write('Strategy,Sharpe,Fitnessvalue,turnover,winlossratio,IC,IR,Rank_IC,Rank_IR,Insample Sharpe,Outsample Sharpe,IS_winlossratio,OS_winlossratio,Long Fitness,Short Fitness,Long Insample Fitness,Short Insample Fitness,Long Outsample Fitness,Short Outsample Fitness,IS_turnover,OS_turnover,IS_IC,IS_IR,OS_IC,OS_IR,IS_Rank_IC,IS_Rank_IR,OS_Rank_IC,OS_Rank_IR')


    print("Start Generate Combinations")
    input_data_list = [input_data for input_data in input_data_list if 'PerTrade' not in input_data]
    # combinations = list(itertools.product(input_data_list,formula_list,['','Reverse']))
    combinations = list(itertools.product(input_data_list,formula_list))
    combinations = [f'{combination[0]}.{combination[1]}'for combination in combinations]
    combinations = list(set(combinations) - set(prev_result))

    print("Generate Combinations completed")


    
    BM = crawl_cmcIndex()['CryptoMarket']
    BM = BM.loc[datetime(2020,1,1):df_data['Close'].index[-1]]
    BM = BM.pct_change()



    # Run
    process_list = []
    # combinations = combinations[:num_process+1]
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
