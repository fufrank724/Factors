import numpy as np
import pandas as pd
import itertools

FORMULAS = ['Pct_Change',#'Shift',
            'Abs','Sign','SignedPower',
            'Std','Skew','Kurt',#'Corr_with_Shift'
            # 'Sin','Cos','Tan','Sinh','Cosh','Tanh','Sigmoid',
            'Mean','Quantile25','Quantile50','Quantile75','Min','Max',
            'Rank',
            # 'Decay_linear'
            ]
FORMULAS_PARAM_FREE = ['Abs','Sign','SignedPower','Sin','Cos','Tan','Sinh','Cosh','Tanh','Sigmoid','Decay_linear']

def factor_filtered(formula):
    operators = formula.split('.')
    conditions = []
    if len(operators) > 1:
        conditions += [
            (operators[0] == 'Abs') and len(operators) > 1,
            (operators[0] == 'Sign') and len(operators) > 1,
            (operators[0] == 'Mean') and (operators[1] == 'Sign'),
            (operators[0] == 'Qauntile') and (operators[1] == 'Sign'),
            (operators[0] == 'Min') and (operators[1] == 'Sign'),
            (operators[0] == 'Max') and (operators[1] == 'Sign'),

            (operators[0] == 'Mean') and (operators[1] == 'Abs'),
            (operators[0] == 'Qauntile25') and (operators[1] == 'Abs'),
            (operators[0] == 'Qauntile50') and (operators[1] == 'Abs'),
            (operators[0] == 'Qauntile75') and (operators[1] == 'Abs'),
            (operators[0] == 'Min') and (operators[1] == 'Abs'),
            (operators[0] == 'Max') and (operators[1] == 'Abs'),

            'Sign.Abs' in formula
        ]

    result_is_not_price = False
    CONVERT_FORMULAS = ['Std','Skew','Kurt','Pct_Change','Rank']

    for i in range(max(len(operators)-1,1)):
        if len(operators) == 1:
            result_is_not_price = result_is_not_price or (operators[i] in CONVERT_FORMULAS) 
        else:
            result_is_not_price = result_is_not_price or ((operators[i] in CONVERT_FORMULAS) or (operators[i+1] in CONVERT_FORMULAS))
            conditions += [
                (operators[i] == 'Abs') and (operators[i+1] == 'Sign'),
                (operators[i] == 'Std') and (operators[i+1] == 'Sign'),
                (operators[i] == 'Cosh') and (operators[i+1] == 'Sign'),
                (operators[i] == 'Rank') and (operators[i+1] == 'Sign'),
                (operators[i] == 'Shift') and (operators[i+1] == 'Shift'),
            ]

    # print(formula,sum(conditions))
    return sum(conditions) == 0 and result_is_not_price
    
def deal_with_shift(formulas):
    for i,formula in enumerate(formulas):
        operators = formula.split('.')
        new_formula = formula.split('.')
        for k,operator in enumerate(operators[:-1]):
            if operators[k] == 'Shift':
                new_formula[k] = operators[k+1]
                new_formula[k+1] = operators[k]
                
        formulas[i] = '.'.join(new_formula)
    return formulas

def calc_factors(input_data,formula,params = None):
    operators = formula.split('.')

    if not (params is None):
        params_combination = params
    else:
        params_combination = list(itertools.product([24*7],repeat = len(operators)))[0]

    factor = input_data.copy()
    i = 0
    for operator in operators:
        if operator in FORMULAS_PARAM_FREE:
            factor = calc_operator(factor,operator,0)
        else:        
            factor = calc_operator(factor,operator,params_combination[i])
            i += 1
    return factor


def calc_operator(data,operator,period):
    if operator == 'Pct_Change':
        return data.pct_change(period)
    elif operator == 'Abs':
        return data.abs()
    elif operator == 'Sign':
        return data.apply(lambda x:np.sign(x))
    elif operator == 'SignedPower':
        return data.apply(lambda x:np.sign(x) * np.power(np.abs(x),np.e))
    elif operator == 'Std':
        return data.rolling(period).std()
    elif operator == 'Skew':
        return data.rolling(period).skew()
    elif operator == 'Kurt':
        return data.rolling(period).kurt()
    elif operator == 'Sin':
        return data.apply(lambda x:np.sin(x))
    elif operator == 'Cos':
        return data.apply(lambda x:np.cos(x))
    elif operator == 'Tan':
        return data.apply(lambda x:np.tan(x))
    elif operator == 'Sinh':
        return data.apply(lambda x:np.sinh(x))
    elif operator == 'Cosh':
        return data.apply(lambda x:np.cosh(x))
    elif operator == 'Tanh':
        return data.apply(lambda x:np.tanh(x))
    elif operator == 'Sigmoid':
        return data.apply(lambda x:np.tanh(x))
    elif operator == 'Mean':
        return data.rolling(period).mean()
    elif operator == 'Quantile25':
        return data.rolling(period).quantile(0.25)
    elif operator == 'Quantile50':
        return data.rolling(period).quantile(0.5)
    elif operator == 'Quantile75':
        return data.rolling(period).quantile(0.75)
    elif operator == 'Min':
        return data.rolling(period).min()
    elif operator == 'Max':
        return data.rolling(period).max()
    elif operator == 'Rank':
        return data.rolling(period).rank(pct = True)
    elif operator == 'Decay_linear':
        return 
    else:
        return

def calc_input_data(df,data_name):
    op_stack = []
    prev_op = 0
    result = 0
    if '*' not in data_name and '/' not in data_name:
        return df[data_name]
    for i,c in enumerate(data_name):
        if c in ['*','/']:
            if len(op_stack)==0:
                result = df[data_name[prev_op:i]].copy()
                op_stack.append(c)
            else:
                if op_stack[0] == '*':
                    result *= df[data_name[prev_op:i]]
                else:
                    result /= df[data_name[prev_op:i]]
                op_stack.clear()
                op_stack.append(c)

            
            prev_op = i + 1
        elif i == len(data_name)-1:            
            if op_stack[0] == '*':
                result *= df[data_name[prev_op:]]
            else:
                result /= df[data_name[prev_op:]]

            op_stack.clear()
            op_stack.append(c)
    return result

