import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import copy
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score as rr2
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

"""
环比变化
"""
factors_df = pd.read_excel('factors_wk_friday.xlsx')
factors_df = factors_df.dropna()
factors_df.index = factors_df.iloc[:, 0]
factors_df = factors_df.drop(columns = ['指标名称'])
factors_df = factors_df.drop(index = ['频率'], axis = 0)
# factor_name = list(factors_df.columns.values)
factors_df = factors_df.diff()
factors_df = factors_df.dropna()
factors_df = factors_df.drop(factors_df.index[-1])
new_columns_list = [column_str + '_diff' for i, column_str in enumerate(factors_df.columns)]
new_columns = pd.core.indexes.base.Index(new_columns_list)
factors_df.columns = new_columns

"""
原因子值
"""
factors_df_2 = pd.read_excel('factors_wk_friday.xlsx')
factors_df_2 = factors_df_2.dropna()
factors_df_2.index = factors_df_2.iloc[:, 0]
factors_df_2 = factors_df_2.drop(columns = ['指标名称'])
factors_df_2 = factors_df_2.drop(index = ['频率'], axis = 0)
factors_df_2 = factors_df_2.drop(factors_df_2.index[0:4])
factors_df_2 = factors_df_2.drop(factors_df_2.index[-1])

"""
turn to weekly (chain) (wind transform not complete)
"""
for c in factors_df.columns:
    for i in range(1, len(factors_df.index)):
        if (factors_df.index[i].month == factors_df.index[i - 1].month) and (factors_df[c][factors_df.index[i:i+4]] == 0).all():
            factors_df[c][factors_df.index[i:i+4]] = factors_df[c][factors_df.index[i - 1]]
        elif (factors_df.index[i].month == factors_df.index[i - 1].month) and (factors_df[c][factors_df.index[i:i+3]] == 0).all():
            factors_df[c][factors_df.index[i:i+3]] = factors_df[c][factors_df.index[i - 1]]
        else:
            pass
factors_df = factors_df.drop(factors_df.index[0:3])

"""
合并因子和环比变化
"""
factors_df = pd.concat([factors_df, factors_df_2], axis = 1)
factor_name = list(factors_df.columns.values)

"""
目前数据是使用这周五得到的宏观因子数据环比变化预测下一周（下周一到下下周一）的十年期国开收益率的变化
!!!一定要注意2019年的国庆10.7刚好是周一，所以读下来的国开收益率数据相对于宏观数据少了一行!!!
"""
# cdb_df = pd.read_excel('gk_mon.xlsx')
# cdb_df.index = cdb_df.iloc[:, 0]
# cdb_df = cdb_df.drop(columns = ['指标名称'])
# cdb_df = cdb_df.drop(index = ['频率'], axis = 0)
# cdb_df = cdb_df.diff()
# cdb_df = cdb_df.dropna()
# cdb_df = cdb_df.drop(cdb_df.index[0:5])
# cdb_df.loc[datetime.datetime(2019, 10, 7, 0, 0)] = 0.0625
# cdb_df = cdb_df.sort_index()

cdb_df = pd.read_excel('gk_mon.xlsx')
cdb_df.index = cdb_df.iloc[:, 0]
cdb_df = cdb_df.drop(columns = ['指标名称'])
cdb_df = cdb_df.drop(index = ['频率'], axis = 0)
cdb_df = cdb_df.dropna()
cdb_df.loc[datetime.datetime(2016, 10, 7, 0, 0)] = 3.0554
cdb_df.loc[datetime.datetime(2020, 1, 31, 0, 0)] = 3.4127
cdb_df = cdb_df.sort_index()
cdb_df = cdb_df.drop(cdb_df.index[-4:-1])

def factor_standard(factor_data):
    factor_data_after_stan = pd.DataFrame()
    for c in factor_data.columns:
        mean = factor_data[c].mean()
        std = factor_data[c].std()
        factor_data_after_stan[c] = (factor_data[c] - mean) / std
    return factor_data_after_stan

def LLT(factor_data, par):
    """
    factor_data: dataset of all factors (predictors)
    par: LLT period parameters
    """
    cp = copy.deepcopy(factor_data)
    tmp = copy.deepcopy(factor_data)
    a = 2.0 / (par + 1.0)
    for c in factor_data.columns:
        for i in np.arange(2, len(tmp)):
            tmp[c][i] = (a - (a*a)/4) * cp[c][i] + ((a*a)/2)*cp[c][i - 1] - (a - 3*(a*a) / 4) * cp[c][i - 2] + 2 * (1 - a) * tmp[c][i - 1] - (1 - a) * (1 - a) * tmp[c][i - 2]
            return tmp
    
def tprf(X, y, Z, oos_present, oos = []):
    """
    X: Array of predictors (T x N, T = number of timestamps, N = number of predictors)
    y: Array of returns (Shape: T x 1, T = number of timestamps)
    Z: Array of proxies (Shape: T x L, T = number of timestamps, L = number of proxies)
    oos_present: True if out of sample data is present (oos), False otherwise
    oos: Out of sample data (predictors)(Shape: 1 x N)
    
    yhat: Forecasted returns for in sample data (Shape: T x 1)
    yhatt: Forecasted return for out of sample array of predictors (float)
    """

    # P1 (Dependent - Value of predictor i across given time intervals,
    # Independent - Set of proxies)
    # phi: rows num = num of factors; cols num = num of proxies
    phi = np.ndarray(shape=(X.shape[1], Z.shape[1]))
    eta = []
    # X.shape[1]: number of factors
    for i in range(0, X.shape[1]):
        first_p_model = LR()
        first_p_model = first_p_model.fit(X = Z, y = X[:, i])
        phi[i, :] = first_p_model.coef_
        eta.append(first_p_model.intercept_)

    # P2 (Dependant - Cross section of predictor values at time t,
    # Independent - phi (from P1)

    eta = np.array(eta).reshape(X.shape[1], 1)
    sigma = np.ndarray(shape = (X.shape[0], Z.shape[1]))
    eta1 = []
    # X.shape[0]: number of timestamps
    for t in range(0, X.shape[0]):
        second_p_model = LR()
        second_p_model.fit(X = phi, y = X[t, :].T)
        sigma[t, :] = second_p_model.coef_.flatten()
        eta1.append(second_p_model.intercept_)
    eta1 = np.array(eta1)

    # P3 (Dependant - Array of returns, Independent - sigma (from P2)

    third_p_model = LR()
    third_p_model.fit(X = sigma, y = y)
    coeff = third_p_model.coef_
    intercept = third_p_model.intercept_
    yhat = np.dot(sigma, coeff.T) + intercept

    # If out of sample set of predictors is present, compute the forecasted
    # return by running the second pass with out of sample predictors as the
    # dependant variable, and multiplying the resultant sigma with coeff from
    # the previous third pass and adding the intercept

    yhatt = np.nan
    if oos_present:
        second_p_model = LR()
        second_p_model.fit(X = phi, y = oos)
        sigma = second_p_model.coef_.flatten()
        yhatt = np.dot(sigma, coeff.T) + intercept
    return yhat, yhatt, sigma

def autoproxy(X, y, n_proxy):
    """
    Use the autoproxy algorithm for calculating the proxies,
    given an array of predictors and corresponding target values

    X: Array of predictors
    y: Array of target values
    n_proxy: number of proxies to be calculated
    """
    r0 = np.array(y)
    yhatt = 1
    for i in range(0, n_proxy - 1):
        (yhat, yhatt, sigma) = tprf(X, y, r0, False)
        r0 = np.hstack([y - yhat, r0])
    return r0
 
def recursive_train(X, y, Z, train_window):
    """
    Recursively train on the training data and predict on the
    out of sample data
    X: Array of predictors
    y: Array of gk returns
    Z: If int: number of proxies to be calculated
    If array: Array of proxies (Shape: TxL)
    train_window: Initial training size to be used
    """
    lst = []
    if isinstance(Z, int):
        do_autoproxy = True
        n_proxies = Z
    else:
        do_autoproxy = False
    for t in range(train_window, X.shape[0]):
        if do_autoproxy: 
            Z = autoproxy(X[t-52:t], y[t-52:t].reshape(-1, 1), n_proxies)
        else:
            Z = Z[t-52:t]
        X_train = X[t-52:t]
        # X_train = (X_train.T/np.std(X_train, axis = 0).reshape(-1, 1)).T
        X_test = X[t]
        y_train = y[t-52:t]
        yhat, yhatt, sigma = tprf(X_train, y_train, Z, True, X_test)
        lst.append(yhatt)
    yhatt = np.array(lst)
    y_true = y[train_window:]
    return rr2(y_true, yhatt)

"""
测试r^2（没有合并前的情况）
"""
r = []
for auto_num in range(1, 21, 1):
    X = factors_df.drop(factors_df.index[:55])
    X = LLT(X, 12)
    X = factor_standard(X).values
    y = cdb_df.drop(cdb_df.index[:55]).values
    # lst = []
    lstt = []
    for t in range(52, X.shape[0]):
        Z = autoproxy(X[t-52:t], y[t-52:t], auto_num)
        X_train = X[t-52:t]
        X_train = np.array(X_train, dtype = np.float64)
        # X_train = (X_train.T/np.std(X_train, axis = 0).reshape(-1, 1)).T
        X_test = X[t]
        # X_test = (X_test.T/np.std(X_test, axis = 0).reshape(-1, 1).flatten()).T
        y_train = y[t-52:t]
        yhat, yhatt, sigma = tprf(X_train, y_train, Z, True, X_test)
        # lst.append(yhat)
        lstt.append(yhatt)
    # yhat = np.array(lst)
    yhatt = np.array(lstt)
    y_true = y[52:]
    # l_yhat = []
    # for i in range(len(y_true)):
    #     l_yhat.append(yhat[i][-1][0])
    r.append(rr2(y_true, yhatt))
    
plt.plot(y_true)
plt.plot(yhatt)

"""
测试（合并后）
"""
r = []
for auto_num in range(1, 21, 1):
    X = factors_df.drop(factors_df.index[:55])
    X = LLT(X, 12)
    X = factor_standard(X).values
    y = cdb_df.drop(cdb_df.index[:55]).values
    # lst = [] 
    lstt = []
    for t in range(52, X.shape[0]):
        Z = autoproxy(X[t-52:t], y[t-52:t], auto_num)
        X_train = X[t-52:t]
        X_train = np.array(X_train, dtype = np.float64)
        # X_train = (X_train.T/np.std(X_train, axis = 0).reshape(-1, 1)).T
        X_test = X[t]
        # X_test = (X_test.T/np.std(X_test, axis = 0).reshape(-1, 1).flatten()).T
        y_train = y[t-52:t]
        yhat, yhatt, sigma = tprf(X_train, y_train, Z, True, X_test)
        # lst.append(yhat)
        lstt.append(yhatt)
    # yhat = np.array(lst)
    yhatt = np.array(lstt)
    y_true = y[52:]
    r.append(rr2(y_true, yhatt))


X = factors_df.drop(factors_df.index[:55])
X = LLT(X, 12)
X = factor_standard(X).values
y = cdb_df.drop(cdb_df.index[:55]).values
# lst = [] 
lstt = []
for t in range(52, X.shape[0]):
    Z = autoproxy(X[t-52:t], y[t-52:t], 17)
    X_train = X[t-52:t]
    X_train = np.array(X_train, dtype = np.float64)
    # X_train = (X_train.T/np.std(X_train, axis = 0).reshape(-1, 1)).T
    X_test = X[t]
    # X_test = (X_test.T/np.std(X_test, axis = 0).reshape(-1, 1).flatten()).T
    y_train = y[t-52:t]
    yhat, yhatt, sigma = tprf(X_train, y_train, Z, True, X_test)
    # lst.append(yhat)
    lstt.append(yhatt)
# yhat = np.array(lst)
yhatt = np.array(lstt)
y_true = y[52:]
rr2(yhatt, y_true)

plt.plot(y_true)
plt.plot(yhatt)

# ls_yhatt = []
# for y in yhatt:
#     ls_yhatt.append(y[0])

# ls_ytrue = []
# for y in y_true:
#     ls_ytrue.append(y[0])

# # exp_index = []
# # for i in range(len(ls_yhatt)):
# #     if abs(yhatt[i]) >= 6:
# #         exp_index.append(i)

# for i in range(len(ls_yhatt)):
#     if abs(yhatt[i]) >= 6:
#         yhatt[i] = yhatt[i - 1]

# final_ytrue = []
# final_yhatt = []
# for i in range(len(ls_ytrue)):
#     if i not in exp_index:
#         final_ytrue.append(ls_ytrue[i])
#         final_yhatt.append(ls_yhatt[i])

# plt.plot(final_ytrue)
# plt.plot(final_yhatt)

