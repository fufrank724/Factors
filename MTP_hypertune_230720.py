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

        long_result = fast_backtest(ret,long_signal,fee = 0)#.sum(axis = 1)
        short_result = fast_backtest(ret,short_signal,fee = 0)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)

        result[result>result.quantile(0.95)] = result.quantile(0.95)

        score = show_performance_metrics(result.loc[datetime(2020,1,1):VALID_END_DATE],show = False)
        IS_score = show_performance_metrics(result.loc[datetime(2020,1,1):INSAMPLE_END_DATE],show = False)#['FitnessValue']
        OS_score = show_performance_metrics(result.loc[INSAMPLE_END_DATE:VALID_END_DATE],show = False)

        IS_score['Sharpe Decay'] = int(abs(OS_score['Ann. Sharpe'] - IS_score['Ann. Sharpe']))
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

        long_result = fast_backtest(ret[subset],long_signal,fee = 4)#.sum(axis = 1)
        short_result = fast_backtest(ret[subset],short_signal,fee = 4)#.sum(axis = 1)

        result = ((long_result + short_result)/2).sum(axis = 1)
        long_result = long_result.sum(axis = 1)
        short_result = short_result.sum(axis = 1)

        result_dict[str(tuple(param)).replace(',','_')] = result

    result = pd.DataFrame(result_dict).loc[datetime(2020,1,1):SAMPLE_END_DATE]
    return result




    
if __name__=='__main__':
    num_process = 30

    INSAMPLE_END_DATE = datetime(2021,12,31)
    VALID_END_DATE = datetime(2022,6,1)
    SAMPLE_END_DATE = datetime(2023,1,1)

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
    df_data['takerBuyQuoteVol'] = df_data['takerBuyQuoteVol'] * 2 - df_data['quoteAssetVolume']



    metrics = pd.read_csv('/home/frank/document/Python/Factors/output/backtest_result/D/level1_IC_metrics2.csv',index_col = 0,dtype = {'Strategy':str})
    metrics = metrics.drop_duplicates('Sharpe')#.sort_values('Fitnessvalue',ascending = False)
    metrics = metrics[metrics['Sharpe'] != metrics['Fitnessvalue']]
    metrics['Reverse'] = 0

    ret = df_data['Close'].sort_index().resample(FREQ).last().bfill().pct_change().fillna(0)

    BM = crawl_cmcIndex()['CryptoMarket']
    BM = BM.loc[datetime(2020,1,1):df_data['Close'].index[-1]]
    BM = BM.pct_change()

    PATH = '/home/frank/document/Python/Factors/output/hypertune_2307'
    if not os.path.isdir(f'{PATH}/'):
        os.mkdir(f'{PATH}/')
    



    for name in metrics.index[:]:
    # for name in ['Close.Pct_Change','Open/Close.SignedPower.Abs.Kurt','Close.Rank']:

        # name = metrics.index[10]
 
        strategy_ix = strategies.index(name.replace('_Reverse',''))

        if metrics['Reverse'].loc[name] == -1:
            name += '_Reverse'


        # if os.path.exists(f"{PATH}/{strategy_ix}/categories_{strategy_ix}.jpg"):
        #     continue

        if not os.path.isdir(f'{PATH}/{strategy_ix}'):
            os.mkdir(f'{PATH}/{strategy_ix}')


        print(name)
        stress_score = {}
        data = name.replace('_Reverse','').split('.')
        formula = '.'.join(data[1:])
        data = data[0]

        operators = formula.split('.')
        
        if 'Sign' in operators:
            continue
        
        vol_len = [24,24*3,24*7,24*14,24*30]
        param_count = sum([1 for operator in operators if operator not in FORMULAS_PARAM_FREE])        
        params = sum([1 for operator in operators if operator not in FORMULAS_PARAM_FREE])
        params = list(itertools.product(list(range(24,30*24 + 8,8)),repeat = params))
        params = list(itertools.product(vol_len,params))
        np.random.shuffle(params)
        params = params[:10**3 * 3]

        FNAME = f'{PATH}/{strategy_ix}/hypertune_First_{strategy_ix}.csv'
        keys = 'Ann. Return(%),MDD(%),Ret/Mdd,Ann. Sharpe,ATH_Score,Monthly_Return_Score,Cummax_Score,FitnessValue,Sharpe Decay'.split(',')
        with open(FNAME,'w') as f:
            f.write(','.join(['vol_len'] + [f'len{i+1}'for i in range(sum([1 for operator in operators if operator not in FORMULAS_PARAM_FREE]))])+',Ann. Return(%),MDD(%),Ret/Mdd,Ann. Sharpe,ATH_Score,Monthly_Return_Score,Cummax_Score,FitnessValue,Sharpe Decay\n')



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
        



        print("Hypertune End")

        # ## Ensamble
        for reverse in [False,True]:
            name += '_Reverse' if reverse else ''

            ENSAMBLE_PATH = f'{PATH}/{strategy_ix}/{reverse}'
            if not os.path.isdir(ENSAMBLE_PATH):
                os.mkdir(ENSAMBLE_PATH)


            
            ix = strategies.index(name.replace('_Reverse',''))
            FNAME = f'{PATH}/{strategy_ix}/hypertune_First_{strategy_ix}.csv'
            second_result = pd.read_csv(FNAME)
        

            second_result['FitnessValue'] = second_result['FitnessValue'].round(1)
            second_result = second_result.sort_values(['FitnessValue','Sharpe Decay'],ascending = [reverse,True])
            second_result = second_result.iloc[:int(second_result.shape[1] * 0.75)]

            total = int(second_result.shape[0]/2)
            partial = int(total/5)
                

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
            selected.to_csv(f'{ENSAMBLE_PATH}/Ensamble_{strategy_ix}.csv')
            selected = selected[selected.columns[:param_count+1]]#.values

            #param stress
            # print(selected.values)
            result = backtest(name,selected.values)
            result.cumsum().plot(title = name)
            result.to_csv(f"{ENSAMBLE_PATH}/{strategy_ix}_Ensamble_nav.csv")
            plt.savefig(f"{ENSAMBLE_PATH}/{strategy_ix}.jpg",dpi = 500)
            plt.close()        

            #selection stress
            selection_result = {}
            for selection in [5,10,20]:
                selection_result[selection] = backtest(name,selected.iloc[:1].values,selected=selection).sum(axis = 1)

            pd.DataFrame(selection_result).cumsum().plot(title = name + f"  {tuple(selected.values[0])}")
            plt.savefig(f"{ENSAMBLE_PATH}/selection_result_{strategy_ix}.jpg",dpi = 500)
            plt.close()     

            #classify stress
            subsets = ['all','token','coin']
            categories_result = {}
            for subset in subsets:
                categories_result[subset] = backtest(name,selected.iloc[:1].values,category=subset).sum(axis = 1)

            pd.DataFrame(categories_result).cumsum().plot(title = name + f"  {tuple(selected.values[0])}")
            plt.savefig(f"{ENSAMBLE_PATH}/categories_{strategy_ix}.jpg",dpi = 500)
            plt.close()        
            print(ix)
        # break

    print('Done')
