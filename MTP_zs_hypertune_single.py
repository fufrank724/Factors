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

def task(strategy,params):
    ## backtest

    data = strategy.replace('_Reverse','').split('.')
    formula = '.'.join(data[1:])
    data = data[0]

    input_data = calc_input_data(df_data,data)

    for param in params:

        factor1 = calc_factors(input_data,formula,param[1])
        factor = factor1.sort_index().resample(FREQ).last()

        # if '_Reverse' in name:
        #     factor *= -1


        market_filter = ~((df_data['quoteAssetVolume'] == 0) | df_data['quoteAssetVolume'].isna())
        volume = df_data['quoteAssetVolume'].sort_index().rolling(param[0]).sum().fillna(0)
        volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
        GLOBAL_FILTER = market_filter & (volume_filter>0.5)
        GLOBAL_FILTER = GLOBAL_FILTER.resample('D').last()
        cond = GLOBAL_FILTER 


        selected = 5
        data = name.replace('_Reverse','').split('.')
        formula = '.'.join(data[1:])
        data = data[0]
        # print(name)

        select = cond.sum(axis = 1) * selected * 0.01
        select = select.apply(lambda x:max(np.floor(x),3))

        rk = factor[cond].rank(axis = 1,ascending = '_Reverse' not in name,method = 'dense')

        long_signal = rk.copy()
        long_signal[:] = 0
        short_signal = long_signal.copy()

        long_signal[rk.gt(rk.max(axis = 1) - select,axis = 0)] = 1 
        short_signal[rk.le(select,axis = 0)] = -1 

        # if '_Reverse' in name:
        #     long_signal *= -1
        #     short_signal *= -1

        long_result = fast_backtest(ret,long_signal)#.sum(axis = 1)
        short_result = fast_backtest(ret,short_signal)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)

        result[result>result.quantile(0.95)] = result.quantile(0.95)

        # score = show_performance_metrics(result.loc[datetime(2020,1,1):SAMPLE_END_DATE],show = False)
        IS_score = show_performance_metrics(result.loc[datetime(2020,1,1):INSAMPLE_END_DATE],show = False)#['FitnessValue']
        # IS_score = show_performance_metrics(result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)
        # OS_score = show_performance_metrics(result.loc[datetime(2020,1,1):SAMPLE_END_DATE],show = False)
        # param = {'len1':param[0],'len2':param[1]}
        hypertune_metrics[tuple([param[0]] + list(param[1]))] = IS_score
        print(strategy,str(tuple([param[0]] + list(param[1]))),IS_score['Ann. Sharpe'])



        msg = ','.join([f'{IS_score[key]:.6f}' for key in keys])
        with open(FNAME,'a') as f:
            f.write(str(tuple([param[0]] + list(param[1])))[1:-1].replace(' ','') + ',' + msg + '\n')



def backtest(strategy,params,selected = 5,category = 'all'):
    ## backtest

    data = strategy.replace('_Reverse','').split('.')
    formula = '.'.join(data[1:])
    data = data[0]

    subset = categories[category]
    if category != 'all':
        subset = np.unique([sym + 'usdt' for sym in subset])

    input_data = calc_input_data(df_data,data)


    result_dict = {}
    for param in params:
        print(formula,param[:])
        factor1 = calc_factors(input_data,formula,param[1:])
        factor = factor1[subset].sort_index().resample(FREQ).last()

        # if '_Reverse' in name:
        #     factor *= -1


        market_filter = ~((df_data['quoteAssetVolume'] == 0) | df_data['quoteAssetVolume'].isna())
        volume = df_data['quoteAssetVolume'].sort_index().rolling(param[0]).sum().fillna(0)
        volume_filter = volume[(market_filter)].rank(axis = 1,pct = True,ascending = True,method = 'dense')
        GLOBAL_FILTER = market_filter & (volume_filter>0.5)
        GLOBAL_FILTER = GLOBAL_FILTER.resample('D').last()
        cond = GLOBAL_FILTER[subset] 


        data = name.replace('_Reverse','').split('.')
        formula = '.'.join(data[1:])
        data = data[0]
        # print(name)

        select = cond.sum(axis = 1) * selected * 0.01
        select = select.apply(lambda x:max(np.floor(x),3))

        rk = factor[cond].rank(axis = 1,ascending = '_Reverse' not in name,method = 'dense')

        long_signal = rk.copy()
        long_signal[:] = 0
        short_signal = long_signal.copy()

        long_signal[rk.gt(rk.max(axis = 1) - select,axis = 0)] = 1 
        short_signal[rk.le(select,axis = 0)] = -1 

        # if '_Reverse' in name:
        #     long_signal *= -1
        #     short_signal *= -1

        long_result = fast_backtest(ret[subset],long_signal)#.sum(axis = 1)
        short_result = fast_backtest(ret[subset],short_signal)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)

        result_dict[tuple(param)] = result

    result = pd.DataFrame(result_dict).loc[datetime(2020,1,1):SAMPLE_END_DATE]
    return result




    
if __name__=='__main__':
    num_process = 30

    INSAMPLE_END_DATE = datetime(2021,12,31)
    VALID_END_DATE = datetime(2022,6,1)
    SAMPLE_END_DATE = datetime(2022,8,31)

    strategies = pd.read_csv('/home/frank/document/Python/Factors/output/backtest_result/D/level1_IC_metrics2.csv').Strategy.sort_values().to_list()

    FREQ = 'D'


    df = pd.read_csv('/home/frank/document/Python/Factors/data/data_1h.csv')
    df['BuyerRatio'] = df['takerBuyQuoteVol']/df['quoteAssetVolume']
    df['BuyerPerTrade'] = df['takerBuyQuoteVol']/df['numberOfTrades']
    df['VolumePerTrade'] = df['quoteAssetVolume']/df['numberOfTrades']

    df = df.drop(['Volume','numberOfTrades'],axis= 1)

    categories = {}
    with open('/home/frank/document/Python/Factors/data/Crytpo_categories.json','r') as f:
        categories = json.loads(f.read())    
    categories['all'] = df.symbol.unique()

    print('Dataframe prepared completed')

    ## Pivot tables
    col_list = ['Open','High','Low','Close','quoteAssetVolume','takerBuyQuoteVol','BuyerRatio']
    df_data = {}
    for col in col_list:
        df_data[col] = df.pivot(values = col,index = 'openTime',columns = 'symbol').astype(float)
        df_data[col].index = pd.to_datetime(df_data[col].index,unit = 'ms') #+ timedelta(hours=8)



    metrics = pd.read_csv('/home/frank/document/Python/Factors/output/backtest_result/D/level1_IC_metrics.csv',index_col = 0,dtype = {'Strategy':str})
    metrics = metrics.sort_index().drop_duplicates('Sharpe').sort_values('Fitnessvalue',ascending = False)
    metrics = metrics[metrics['Sharpe'] != metrics['Fitnessvalue']]
    metrics['Reverse'] = 0
    metrics['Reverse'][metrics['Sharpe']<0] = -1


    ret = df_data['Close'].sort_index().resample(FREQ).last().bfill().pct_change().fillna(0)

    BM = crawl_cmcIndex()['CryptoMarket']
    BM = BM.loc[datetime(2020,1,1):df_data['Close'].index[-1]]
    BM = BM.pct_change()

    PATH = f'/home/frank/document/Python/Factors/output/hypertune_individual/{int(datetime.now().timestamp())}'
    if not os.path.isdir(f'{PATH}/'):
        os.mkdir(f'{PATH}/')

    # for name in metrics.index[:100]:
    for name in ['High/Close.Pct_Change.Max.Skew_Reverse']:
        # name = metrics.index[10]
        # name = 'Open/Close.SignedPower.Abs.Kurt_Reverse'
        # name = 'Low/High.SignedPower.Std.Skew'
        ix = strategies.index(name.replace('_Reverse',''))

        # if metrics['Reverse'].loc[name] == -1:
        #     name += '_Reverse'


        # if os.path.exists(f"{PATH}/categories_{ix}.jpg"):
        #     continue

        if not os.path.isdir(f'{PATH}'):
            os.mkdir(f'{PATH}')


        print(name)
        stress_score = {}
        data = name.replace('_Reverse','').split('.')
        formula = '.'.join(data[1:])
        data = data[0]

        operators = formula.split('.')
        vol_len = [24,24*3,24*7,24*14,24*30]
        params = sum([1 for operator in operators if operator not in FORMULAS_PARAM_FREE])
        # params = list(itertools.product(list(range(24,30*24 + 8,8)),repeat = params))
        params = list(itertools.product(list(range(24,200,8)),list(range(24,150,8)),list(range(300,500,8))))
        params = list(itertools.product(vol_len,params))
        np.random.shuffle(params)
        params = params[:10**3 * 3]

        FNAME = f'{PATH}/hypertune_First_{ix}.csv'
        keys = 'Ann. Return(%),MDD(%),Ret/Mdd,Ann. Sharpe,ATH_Score,Monthly_Return_Score,Cummax_Score,FitnessValue'.split(',')
        with open(FNAME,'w') as f:
            f.write(','.join(['vol_len'] + [f'len{i+1}'for i in range(sum([1 for operator in operators if operator not in FORMULAS_PARAM_FREE]))])+',Net_profit(%),MDD(%),Profit_mdd,Sharpe,ATH_Score,Monthly_Return_Score,Cummax_Score,FitnessValue\n')



        hypertune_metrics = {}
        # # Run

        ## First
        process_list = []
        ix = int(len(params[:])/num_process)
        for i in range(num_process):
            process_list.append(mp.Process(target = task, args = (name,params[i*ix:(i+1)*ix],)))
            process_list[i].start()
        i += 1
        process_list.append(mp.Process(target = task, args = (name,params[i*ix:],)))
        process_list[i].start()


        for i in range(len(process_list)):
            process_list[i].join()
        print("\nFirst Hypertune End\n")
        
        # First Ensamble
        # pass


        ## Second
        # hypertune_metrics = {}
        # first_result = pd.read_csv(FNAME)
        # first_result = first_result[(first_result['FitnessValue']>0) & (first_result['FitnessValue']<first_result.FitnessValue.drop_duplicates().quantile(0.99))].sort_values('FitnessValue',ascending = False)#.drop_duplicates('Sharpe')
        # first_result = first_result[(first_result['FitnessValue']>=first_result['FitnessValue'].drop_duplicates().quantile(0.75))]
        # if first_result.size == 0:
        #     continue


        # ## Ensamble
        second_result = pd.read_csv(FNAME)
        # second_result = second_result[(second_result['FitnessValue']>0) & (second_result['FitnessValue']<second_result.FitnessValue.drop_duplicates().quantile(0.99))].sort_values('FitnessValue')#.drop_duplicates('Sharpe')
        # second_result = second_result[(second_result['FitnessValue']<second_result.FitnessValue.drop_duplicates().quantile(0.99))]#.sort_values('FitnessValue')#.drop_duplicates('Sharpe')
        second_result = second_result[(second_result['FitnessValue']>=second_result['FitnessValue'].drop_duplicates().quantile(0.75))]
        second_result = second_result.sort_values('FitnessValue',ascending = False)


        
        total = int(second_result.shape[0]/2)
        partial = int(total/5)
        param_count = sum([1 for operator in operators if operator not in FORMULAS_PARAM_FREE])

        ls = []
        if partial <= 2:
            selected = second_result.copy()
            selected = selected.iloc[np.random.randint(0,selected.shape[0],size = max(selected.shape[1],10))]
            ls.append(selected)
        else:
            for i in range(min(partial,5)):
                # print(i)
                selected = second_result.iloc[second_result.shape[0] - partial * (i+1) : second_result.shape[0] - partial * (i)]
                selected = selected.iloc[np.random.randint(0,selected.shape[0],size = 2)]
                ls.append(selected)
        selected = pd.concat(ls,axis = 0).sort_values('FitnessValue',ascending = False)
        selected.to_csv(f'{PATH}/Ensamble_{ix}.csv')
        selected = selected[selected.columns[:param_count+1]]#.values

        #param stress
        # print(selected.values)
        result = backtest(name,selected.values)
        result.cumsum().plot(title = name)
        result.to_csv(f"{PATH}/Ensamble_nav.csv")
        plt.savefig(f"{PATH}/Ensamble_nav.jpg",dpi = 500)
        plt.close()        

        #selection stress
        selection_result = {}
        for selection in [5,10,20]:
            selection_result[selection] = backtest(name,selected.iloc[:1].values,selected=selection).sum(axis = 1)

        pd.DataFrame(selection_result).cumsum().plot(title = name + f"  {tuple(selected.values[0])}")
        plt.savefig(f"{PATH}/selection_resul.jpg",dpi = 500)
        plt.close()     

        #classify stress
        subsets = ['all','token','coin','Ethereum','bnb-chain']
        categories_result = {}
        for subset in subsets:
            categories_result[subset] = backtest(name,selected.iloc[:1].values,category=subset).sum(axis = 1)

        pd.DataFrame(categories_result).cumsum().plot(title = name + f"  {tuple(selected.values[0])}")
        plt.savefig(f"{PATH}/categories.jpg",dpi = 500)
        plt.close()        

        # break

    print('Done')
