"""
DB GAPS 4회 AlphaLab 전략

모멘텀 스코어 기반
"""

# It requires kiwooma package which is python wrapper module for kiwoom OPEN API made by me

from kiwooma import EasyAPI
from collections import OrderedDict
from scipy.optimize import minimize
import time, datetime
import pandas as pd
import numpy as np

stock_weights = np.zeros(6)
bond_weights = np.zeros(3)

def maximize_score(scores, cov):

    def obj_func(w):
        # total score 최대화 함수
        w = w.reshape(len(scores), 1)
        new_scores = scores.copy()
        # new_scores.iloc[0:6] = scores.iloc[0:6].mean()
        # new_scores.iloc[6:9] = scores.iloc[6:9].mean()
        total_score = -(w.T).dot(new_scores)
        return total_score

    def vol(w):
        return (w.T.dot(cov).dot(w) * 20)**(1/2)

    def allocate_stocks(w):
        global stock_weights
        new_scores = scores.iloc[0:6].copy()
        new_scores[new_scores<0] = 0.0
        weights = np.array(new_scores/new_scores.sum())*0.4
        stock_weights = weights
        return weights

    def allocate_bonds(w):
        global bond_weights
        new_scores = scores.iloc[6:9].copy()
        new_scores[new_scores<0] = 0.0
        weights = np.array(new_scores/new_scores.sum())*0.6
        bond_weights = weights
        return weights

    cons = ({
                'type': 'ineq',
                'fun': lambda w: 0.99 - w.sum()
            },
            {
                'type': 'ineq',
                'fun': lambda w: w - 0.0000001
            },
            # {
            #     'type': 'ineq',
            #     'fun': lambda w: 0.005 - vol(w)
            # },
            {   # Kodex200
                'type': 'ineq',
                'fun': lambda w: 0.4 - w[0] 
            },
            {   # Kosdaq150
                'type': 'ineq',
                'fun': lambda w: 0.4 - w[1]
            },
            {   # Kodex200 & Kosdaq150
                'type': 'ineq',
                'fun': lambda w: (w[0] + w[1]) - 0.1
            },
            {   # Kodex200 & Kosdaq150
                'type': 'ineq',
                'fun': lambda w: 0.4 - (w[0] + w[1])
            },
            {   # 해외주식
                'type': 'ineq',
                'fun': lambda w: (w[2] + w[3] + w[4] + w[5]) - 0.1
            },
            {   # 해외주식
                'type': 'ineq',
                'fun': lambda w: 0.4 - (w[2] + w[3] + w[4] + w[5])
            },
            {   # 국채 10년
                'type': 'ineq',
                'fun': lambda w: 0.5 - w[6]
            },
            {   # 우량회사채
                'type': 'ineq',
                'fun': lambda w: 0.4 - w[7]
            },
            {   # 해외채권
                'type': 'ineq',
                'fun': lambda w: 0.4 - w[8]
            },
            {   # 해외채권
                'type': 'ineq',
                'fun': lambda w: w[8] - 0.05
            },
            {   # 채권
                'type': 'ineq',
                'fun': lambda w: 0.6 - (w[6] + w[7] + w[8])
            },
            {   # 채권
                'type': 'ineq',
                'fun': lambda w: (w[6] + w[7] + w[8]) - 0.2
            },
            {   # 채권
                'type': 'ineq',
                'fun': lambda w: allocate_bonds(w) - w[6:9]
            },
            {   # 금
                'type': 'ineq',
                'fun': lambda w: 0.15 - w[9] 
            },
            {   # WTI
                'type': 'ineq',
                'fun': lambda w: 0.15 - w[10]
            },
            {   # 원자재 
                'type': 'ineq',
                'fun': lambda w: 0.20 - (w[9] + w[10])
            },
            {   # 원자재 
                'type': 'ineq',
                'fun': lambda w: (w[9] + w[10]) - 0.05
            },
            {   # KOSPI Inverse 
                'type': 'ineq',
                'fun': lambda w: w[11]
            },
            {   # KOSPI Inverse 
                'type': 'ineq',
                'fun': lambda w: (w[0]+w[1])-w[11]
            },
            {   # Dollar  
                'type': 'ineq',
                'fun': lambda w: 0.2 - w[12]
            },
            {   # Dollar Inverse 
                'type': 'ineq',
                'fun': lambda w: 0.2 - w[13]
            },
            {   # Dollar Inverse 
                'type': 'ineq',
                'fun': lambda w: 0.2 - (w[12] + w[13])
            },
            {   # 단기자금
                'type': 'ineq',
                'fun': lambda w: 0.5 - w[14]
            },
            {   # 주식
                'type': 'ineq',
                'fun': lambda w: allocate_stocks(w) - w[0:6]
            },)

    #First guess: 동일가중
    w0 = np.ones(len(scores))/len(scores) 

    res = minimize(obj_func, w0, constraints=cons, method = 'SLSQP') #Minimization
    res = (pd.Series(res['x'], index=scores.index)*100).round(4)

    global_stock = res.iloc[2:6].sum()
    bonds = res.iloc[6:9].sum()

    res.loc['S&P500'] = global_stock * stock_weights[2]/stock_weights[2:].sum()
    res.loc['EuroStoxx'] = global_stock * stock_weights[3]/stock_weights[2:].sum()
    res.loc['Nikkei'] = global_stock * stock_weights[4]/stock_weights[2:].sum()
    res.loc['CSI300'] = global_stock * stock_weights[5]/stock_weights[2:].sum()
    res.loc['국고채10년'] = bonds * bond_weights[0]/bond_weights.sum()
    res.loc['중기우량회사채'] = bonds * bond_weights[1]/bond_weights.sum()
    res.loc['하이일드'] = bonds * bond_weights[2]/bond_weights.sum()
    vol = vol(np.array(res))
    res['현금'] = 100-res.sum()
    return res, vol

def calc_scores(code, price):
    if code == '132030':
        returns_20 = price.iloc[-1]/price.iloc[-21]-1
        returns_40 = price.iloc[-1]/price.iloc[-41]-1
        score = (-returns_20 - ((1+returns_40)**0.5-1))/2
    elif code == '138230' or code == '139660':
        returns_40 = price.iloc[-1]/price.iloc[-41]-1
        returns_80 = price.iloc[-1]/price.iloc[-81]-1
        score = ((1+returns_40)**(1/2) + (1+returns_80)**(1/4)-2)/2

    elif code ==  '195930':
        score = 0
    else:
        returns_40 = price.iloc[-1]/price.iloc[-41]-1
        returns_80 = price.iloc[-1]/price.iloc[-81]-1
        returns_120 = price.iloc[-1]/price.iloc[-121]-1
        returns_160 = price.iloc[-1]/price.iloc[-161]-1
        score = ((1+returns_40)**(1/2) + (1+returns_80)**(1/4) + \
            (1+returns_120)**(1/6) + (1+returns_160)**(1/8)-4)/4

    return score

def get_data(api, code):
    df = api.get_daily_ohlcv(code, adj_close=1, repeat=1)
    price_df = df[['open', 'high', 'low', 'close']].mean(1)
    return price_df

def get_scores(price_dict, codes, iloc):
    scores = {}
    dfs = []

    for code in codes:
        price = price_dict[code].iloc[:iloc]
        
        returns = price.pct_change()
        dfs.append(returns)
        returns.name = code
        score = calc_scores(code, price)
        scores[code] = score
        time.sleep(0.2)
    df = pd.concat(dfs, axis=1)
    return scores, df
        
def main():

    stock_dict = OrderedDict({
        '069500': 'KODEX200', '232080': 'KOSDAQ150', '143850': 'S&P500',
        '195930': 'EuroStoxx', '238720': 'Nikkei', '192090': 'CSI300', 
        '148070': '국고채10년', '136340': '중기우량회사채', '182490': '하이일드',
        '132030': '골드선물', '130680': '원유선물', '114800': 'KODEX인버스',
        '138230': '달러선물', '139660': '달러인버스', '130730': '단기자금'
    })
    api = EasyAPI()

    price_dict = {}
    for code in list(stock_dict.keys()):
        price_dict[code] = get_data(api, code)

    scores, df = get_scores(price_dict, list(stock_dict.keys()), -1)

    df.rename(columns = stock_dict, inplace=True)
    cov = df.iloc[-60:].cov()

    new_scores = {}

    for key, item in scores.items():
        new_scores[stock_dict[key]] = item

    ret, vol = maximize_score(pd.Series(new_scores), cov)

    # ret.loc['']
    date = datetime.datetime.today().date()
    with open('{} - scores.txt'.format(date), mode='a', encoding='utf-8') as f:
        f.write('Scores\n-----\n')
        f.write(str(new_scores))
        f.write('\nOur Scores\n-----\n')
        f.write(str(pd.Series(new_scores)*100))
        f.write('\n')
        f.write('Weights\n-----\n')
        f.write('\n')
        f.write(str(ret))
        f.write('\n')
        f.write('Volatiltiy\n-----\n')
        f.write(str(vol))
        f.write('\n')

if __name__ == '__main__':
    main()