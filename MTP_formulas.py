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

def task(num,formulas):
    data_name = 'Close'
    input_data = calc_input_data(df_data,data_name)
    for formula in formulas:

        operators = formula.split('.')
        if operators[-1] == 'Sign':
            continue
        if f'{data_name}.{formula}' in prev_result:
            continue
    
        print(f"{data_name}: {formula}")


        factor = calc_factors(input_data,formula)
        factor = factor.sort_index().resample(FREQ).last()

        cond = GLOBAL_FILTER 

        rk = factor[cond].rank(axis = 1,pct = True,ascending = True,method = 'dense')
        rk = (rk*10).apply(lambda x: np.floor(x))
        rk[rk==10] -= 1


        long_signal = rk.copy()
        long_signal[:] = 0
        short_signal = long_signal.copy()

        long_signal[rk==9] = 1 
        short_signal[rk==0] = -1 




        long_result = fast_backtest(ret,long_signal)#.sum(axis = 1)
        short_result = fast_backtest(ret,short_signal)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)


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

        x = long_result.loc[datetime(2021,1,1):INSAMPLE_END_DATE]/10**4
        y = BM.loc[datetime(2021,1,1):INSAMPLE_END_DATE]
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        long_alpha,long_beta = model.params
        long_adj_rsquare = model.rsquared_adj

        x = short_result.loc[datetime(2021,1,1):INSAMPLE_END_DATE]/10**4
        y = BM.loc[datetime(2021,1,1):INSAMPLE_END_DATE]
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        short_alpha,short_beta = model.params
        short_adj_rsquare = model.rsquared_adj


        msg = f'{data_name}.{formula},{all_sample_result["Sharpe"]:.6f},{all_sample_result["FitnessValue"]:.6f},{insample_result["Sharpe"]:.6f},{outsample_result["Sharpe"]:.6f},{long_all_sample_result["FitnessValue"]:.6f},{short_all_sample_result["FitnessValue"]:.6f},{long_insample_result["FitnessValue"]:.6f},{short_insample_result["FitnessValue"]:.6f},{long_outsample_result["FitnessValue"]:.6f},{short_outsample_result["FitnessValue"]:.6f},{long_alpha:.8f},{long_beta:.8f},{long_adj_rsquare:.8f},{short_alpha:.8f},{short_beta:.8f},{short_adj_rsquare:.8f}'

        with open(fname,'a') as f:
            f.write(msg+'\n')




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
    col_list = ['Close','quoteAssetVolume','takerBuyQuoteVol','BuyerRatio']
    df_data = {}
    for col in col_list:
        df_data[col] = df.pivot(values = col,index = 'openTime',columns = 'symbol').astype(float)
        df_data[col].index = pd.to_datetime(df_data[col].index,unit = 'ms') #+ timedelta(hours=8)

    market_filter = ~((df_data['quoteAssetVolume'] == 0) | df_data['quoteAssetVolume'].isna())
    volume = df_data['quoteAssetVolume'].sort_index().rolling(7*24).sum().fillna(0)
    volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
    GLOBAL_FILTER = market_filter & (volume_filter>0.5)
    GLOBAL_FILTER = GLOBAL_FILTER.resample('D').last()


    print("Pivot table completed")

    ret = df_data['Close'].sort_index().resample(FREQ).last().bfill().pct_change().fillna(0)



    input_data_list = ['High','Low','Close','takerBuyQuoteVol','BuyerRatio']
    formula_list = []
    backtest_metrics = {}

    fname = '/home/frank/document/Python/Factors/output/backtest_result/D/backtest_metrics_formulas.csv'
    

    formula_list = FORMULAS.copy()
    formula_list += ['.'.join(data) for data in list(itertools.product(FORMULAS,repeat = 2))]
    formula_list += ['.'.join(data) for data in list(itertools.product(FORMULAS,repeat = 3))]
    formula_list = [data for data in formula_list if factor_filtered(data)]
    np.random.shuffle(formula_list)


    

    BM = crawl_cmcIndex()['CryptoMarket']
    BM = BM.loc[datetime(2021,1,1):df_data['Close'].index[-1]]
    BM = BM.pct_change()




    ## Prepare Metric File
    prev_result = []
    dt = datetime.today().date().strftime('%Y%m%d')
    if os.path.exists(fname):
        print('file exits')
        prev_result = pd.read_csv(fname).Strategy.to_list()
    else:        
        with open(fname,'w') as f:
            f.write('Strategy,Sharpe,Fitnessvalue,Insample Sharpe,Outsample Sharpe,Long Fitness,Short Fitness,Long Insample Fitness,Short Insample Fitness,Long Outsample Fitness,Short Outsample Fitness,Insample Long Alpha,Insample Long Beta,Insample Long R^2,Insample Short Alpha,Insample Short Beta,Insample Short R^2\n')
    ## Run
    process_list = []
    ix = int(len(formula_list[:])/num_process)
    for i in range(num_process):
        process_list.append(mp.Process(target = task, args = (i,formula_list[i*ix:(i+1)*ix],)))
        process_list[i].start()
    i += 1
    process_list.append(mp.Process(target = task, args = (i,formula_list[i*ix:],)))
    process_list[i].start()


    for i in range(num_process):
        process_list[i].join()
    process_list[-1].join()

    print('Done')
