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


warnings.filterwarnings("ignore")

def task(num,combination_list):
    ## backtest
    backtest_metrics = {}
    backtest_performance = {}


    for combination in combination_list:
        print(combination)
        content = combination.split(',')

        data = content[1].replace('_Reverse','').split('.')
        formula = '.'.join(data[1:])
        data = data[0]
        factor2 = calc_factors(calc_input_data(df_data,data),formula).sort_index().resample(FREQ).last()
        rk2 = factor2.rank(axis = 1,method = 'dense')

        operator = content[2]

        if operator== '*':
            rk = rk1 * rk2
        else:
            rk = rk1 / rk2


        # continue
        cond = GLOBAL_FILTER 

        rk = rk[cond].rank(axis = 1,pct=True,method = 'dense')
        long_signal = rk.add(-rk.mean(axis = 1),axis= 0)
        long_signal[long_signal<0] = 0
        short_signal = rk.add(-rk.mean(axis = 1),axis= 0)
        short_signal[short_signal>0] = 0




        FEE = 0

        long_result = fast_backtest(ret,long_signal,fee = FEE)#.sum(axis = 1)
        short_result = fast_backtest(ret,short_signal,fee = FEE)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)


        ## Performance metrics
        all_sample_result = show_performance_metrics(result.loc[datetime(2021,1,1):SAMPLE_END_DATE],show = False)

        correlation = factor1['ethusdt'].corr(factor2['ethusdt'])

        winloss_ratio = -result[result>0].quantile(0.5)/result[result<0].quantile(0.5)        
        # IS_winloss_ratio = -result[result>0].loc[datetime(2021,1,1):INSAMPLE_END_DATE].quantile(0.5)/result[result<0].loc[datetime(2021,1,1):INSAMPLE_END_DATE].quantile(0.5)
        # OS_winloss_ratio = -result[result>0].loc[INSAMPLE_END_DATE:VALID_END_DATE].quantile(0.5)/result[result<0].loc[INSAMPLE_END_DATE:VALID_END_DATE].quantile(0.5)

        long_signal = long_signal.div(long_signal.sum(axis = 1),axis = 0)
        short_signal = short_signal.div(short_signal.sum(axis = 1),axis = 0) * -1
        signal = (long_signal + short_signal)/2
        turnover = signal.diff().abs().sum(axis = 1)        
        turnover = turnover.loc[datetime(2021,1,1):SAMPLE_END_DATE].mean()

        Rank_IC = 0
        Rank_IR = 0


        msg = f'{combination},{correlation:.4f},{parent_result["Sharpe"]:.6f},{parent_result["FitnessValue"]:.6f},{parent_turnover},{parent_winloss_ratio},{all_sample_result["Sharpe"]:.6f},{all_sample_result["FitnessValue"]:.6f},{turnover:.4f},{winloss_ratio:.4f},{Rank_IC},{Rank_IR}'

        with open(FNAME,'a') as f:
            f.write(msg+'\n')
        # return




if __name__=='__main__':
    num_process = 30

    INSAMPLE_END_DATE = datetime(2021,7,1)
    VALID_END_DATE = datetime(2022,6,1)
    SAMPLE_END_DATE = datetime(2022,8,31)

    FREQ = 'D'


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

    test_data = {}
    for key in df_data.keys():
        test_data[key] = df_data[key]['ethusdt'].copy()


    print("Pivot table completed")

    ## 

    ret = df_data['Close'].sort_index().resample(FREQ).last().bfill().pct_change().fillna(0)


    ## Get input data and formulas
    # input_data_list = []
    # with open('/home/frank/document/Python/Factors/data/data_list/all_data_list_20230221.csv','r') as f:
    #     input_data_list += f.read().split('\n')
    # input_data_list = [input_data for input_data in input_data_list if 'PerTrade' not in input_data]

    # formula_list =  []
    # with open('/home/frank/document/Python/Factors/data/formulas/all_formula_list_20230221.csv','r') as f:
    #     formula_list += f.read().split('\n')

    # formula_list = [formula for formula in formula_list if len(formula.split('.'))<=2]
    # # formula_list = list(itertools.product(formula_list[:],['','Reverse']))

    # data_combinations = list(itertools.product(input_data_list[:],repeat = 2))        
    # formula_combinations = list(itertools.product(formula_list[:],repeat = 2))
    
    metrics = pd.read_csv('/home/frank/document/Python/Factors/output/backtest_result/D/level1_IC_metrics.csv',index_col = 0,dtype = {'Strategy':str})
    metrics = metrics.sort_index().drop_duplicates('Sharpe').sort_values('Fitnessvalue',ascending = False)
    metrics = metrics[metrics['Sharpe'] != metrics['Fitnessvalue']]
    metrics['Reverse'] = 0
    metrics['Reverse'][metrics['Sharpe']<0] = -1

    metrics = metrics[metrics['Insample Sharpe'].abs().round() >=1]

    metrics[['Sharpe']] = metrics[['Sharpe']].abs().round()
    metrics[['Insample Sharpe','Outsample Sharpe']] = metrics[['Insample Sharpe','Outsample Sharpe']].round(2)
    metrics[['IS_Rank_IC','IS_Rank_IR','OS_Rank_IC','OS_Rank_IR']] = metrics[['IS_Rank_IC','IS_Rank_IR','OS_Rank_IC','OS_Rank_IR']].round(3)
    metrics['Sharpe_Decay'] = (metrics['Outsample Sharpe']- metrics['Insample Sharpe']).abs().round(1)
    metrics['turnover'] = metrics['turnover'].round()

    metrics = metrics.sort_values(['Sharpe_Decay','Sharpe','turnover'],ascending = [True,False,True])

    strategies = metrics.index.to_list()
    strategies_operate = metrics.index.to_list()



    ## Global Filter
    market_filter = ~((df_data['quoteAssetVolume'] == 0) | df_data['quoteAssetVolume'].isna())
    volume = df_data['quoteAssetVolume'].sort_index().rolling(7*24).sum().fillna(0)
    volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
    GLOBAL_FILTER = market_filter & (volume_filter>0.5) #& (df_data['quoteAssetVolume'].sort_index().rolling(24*7).sum().fillna(0)/7 > 100*10**6)
    GLOBAL_FILTER = GLOBAL_FILTER.resample('D').last()
    
    BM = crawl_cmcIndex()['CryptoMarket']
    BM = BM.loc[datetime(2021,1,1):df_data['Close'].index[-1]]
    BM = BM.pct_change()


    # Prepare Metric File
    dt = datetime.today().date().strftime('%Y%m%d')
    FNAME = f'/home/frank/document/Python/Factors/output/backtest_result/D/level2_metrics.csv'
    prev_result = []
    if os.path.exists(FNAME):
        print('file exists')
        with open(FNAME,'r') as f:
            prev_result += f.read().split('\n')
            prev_result = [','.join(data.split(',')[:3]) for data in prev_result]

        
    else:        
        with open(FNAME,'w') as f:

            f.write('data1,data2,operator,correlation,Parent_Sharpe,Parent_Fitnessvalue,Parent_turnover,Parent_winloss_ratio,Sharpe,Fitnessvalue,turnover,winloss_ratio,Rank_IC,Rank_IR\n')



    # Run
    for i,strategy in enumerate(strategies):

        data_combinations = list(itertools.product([strategy],strategies[i+1:],['*','/']))
        
        combinations = [f'{data_combination[0]},{data_combination[1]},{data_combination[2]}' for data_combination in data_combinations]

        combinations = list(set(combinations) - set(prev_result))
        print(strategy)
        data = strategy.replace('_Reverse','').split('.')
        formula = '.'.join(data[1:])
        data = data[0]
        factor1 = calc_factors(calc_input_data(df_data,data),formula).resample(FREQ).last()
        rk1 = factor1.rank(axis = 1,method = 'dense')

        cond = GLOBAL_FILTER.copy()
        rk = (rk1)[cond].rank(axis = 1,pct=True,method = 'dense',ascending = False)

        long_signal = rk.add(-rk.mean(axis = 1),axis= 0)
        long_signal[long_signal<0] = 0
        short_signal = rk.add(-rk.mean(axis = 1),axis= 0)
        short_signal[short_signal>0] = 0

        FEE = 0

        long_result = fast_backtest(ret,long_signal,fee = FEE)#.sum(axis = 1)
        short_result = fast_backtest(ret,short_signal,fee = FEE)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)

        parent_result = show_performance_metrics(result.loc[datetime(2021,1,1):SAMPLE_END_DATE],show = False)

        parent_winloss_ratio = -result[result>0].quantile(0.5)/result[result<0].quantile(0.5)        
        parent_IS_winloss_ratio = -result[result>0].loc[datetime(2021,1,1):INSAMPLE_END_DATE].quantile(0.5)/result[result<0].loc[datetime(2021,1,1):INSAMPLE_END_DATE].quantile(0.5)
        parent_OS_winloss_ratio = -result[result>0].loc[INSAMPLE_END_DATE:VALID_END_DATE].quantile(0.5)/result[result<0].loc[INSAMPLE_END_DATE:VALID_END_DATE].quantile(0.5)


        long_signal = long_signal.div(long_signal.sum(axis = 1),axis = 0)
        short_signal = short_signal.div(short_signal.sum(axis = 1),axis = 0) * -1

        signal = (long_signal + short_signal)/2
        turnover = signal.diff().abs().sum(axis = 1)        
        parent_turnover = turnover.loc[datetime(2021,1,1):SAMPLE_END_DATE].mean()

        # print(combinations[:5])
        # break
        process_list = []
        ix = int(len(combinations[:])/num_process)
        for i in range(num_process):
            process_list.append(mp.Process(target = task, args = (i,combinations[i*ix:(i+1)*ix],)))
            process_list[i].start()
        i += 1
        process_list.append(mp.Process(target = task, args = (i,combinations[i*ix:],)))
        process_list[i].start()

        for i in range(len(process_list)):
            process_list[i].join()

        break


    print('Done')
