from datetime import datetime, timedelta
from multiprocessing import Pool

import pandas as pd
import numpy as np
import dateutil
import time
import os

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import ticker
import math

from scipy import stats


class Context:
    def __init__(self, start_date, end_date, group_num, frequency, Path_factor, Path_trade_day, Path_price):
        self.start_date = start_date  # 回测区间
        self.end_date = end_date
        self.td = None  # 此时日期：%Y-%m-%d
        self.N = None  # 资产数目
        self.group_num = group_num  # 因子组数
        self.freq = frequency  # 调仓频率:日频'd',周频'w',月频'm',财报公布期(5,9,11月)'f',每n个交易日n(int)
        self.pos_matrix = None  # 仓位矩阵
        self.net_value = None  # 各组+基准净值
        self.net_value_left = None  # 当前各组合剩余净值
        self.last_td_mark = None
        self.last_day = None
        self.history = {}  # 历史换仓数据：换仓时点、换仓次数、IC、Rank_IC、因子值、价格、股票池、仓位

        self.Path_trade_day = Path_trade_day
        self.trade_day = None  # 交易日数据
        self.Path_factor = Path_factor
        self.factor = None  # 因子数据
        self.Path_price = Path_price
        self.price = None  # 股票后复权价格数据

        self.industry = pd.read_csv("industry申万三级.csv", encoding='gbk')
        self.industry_list = list(set(self.industry["industry"]))
        self.indu = None

        self.industry_value_list = pd.DataFrame()
        self.codes = []
        self.need_rebalance = 0

        self.form = None
        self.codename = None
        # self.group_name=



class Factor_Industry:
    def __init__(self, factor_list):
        # 每个factor_list为一个表格
        for key in factor_list:
            tmp = pd.DataFrame()
            setattr(self, key, tmp)

        # 储存分组收益率结果
        self.total = pd.DataFrame(index=factor_list, columns=['group' + str(i) for i in range(1, 6)] + ['benchmark'])
        self.form_indu = None
        self.form = None


def initialize(context):
    # 交易日信息
    context.trade_day = pd.read_csv(context.Path_trade_day, parse_dates=[0], index_col=[0])
    context.trade_day = context.trade_day[context.start_date:context.end_date].index

    # 价格信息
    context.price = pd.read_csv(context.Path_price, index_col=0, parse_dates=[0])  # parse_date参数：解析第0列为日期格式
    context.price = context.price.loc[context.start_date:context.end_date, context.codes]
    context.price.columns = [x for x in range(1, len(context.price.columns) + 1)]

    # 因子信息，“日期*行业”形式
    # print(context.codes)
    context.factor = pd.read_csv(context.Path_factor, index_col=[0], parse_dates=[0])
    context.factor = context.factor.loc[context.start_date:context.end_date, context.codes]

    # if isinstance(context.freq, int):
    #     context.last_td_mark = 0
    context.last_td_mark = 0
    context.N = context.factor.shape[1]  # N为资产数目
    # 资产矩阵初始化
    group_col = ['group' + str(i + 1) for i in range(context.group_num)]
    group_col.append('benchmark')
    context.net_value = pd.DataFrame(index=context.trade_day, columns=group_col)
    context.net_value.iloc[0, :] = 1  # 设置初始各组资产为1
    # 历史换仓数据
    context.history = {'td': [], 'times': 0, 'IC': [], 'Rank_IC': [], 'factor': [], 'price': [], 'position': []}
    context.stock_position = []


def rebalance(context):
    # 用均值填充NaN
    f = context.factor.loc[context.td, :]

    f[pd.isna(f)] = f.mean()
    f_rank = f.rank(method='first', ascending=False).values  # 使用rank排序，防止组间分布不均
    # 计算权重矩阵
    context.pos_matrix = np.zeros((context.N, context.group_num + 1))

    # print(f_rank)
    for g in range(context.group_num):
        V_min = np.percentile(f_rank, 100 * g / context.group_num, interpolation='linear')
        V_max = np.percentile(f_rank, 100 * (g + 1) / context.group_num, interpolation='linear')
        if g + 1 == context.group_num:
            context.pos_matrix[:, g][(f_rank >= V_min) & (f_rank <= V_max)] = context.net_value_left[g]
        else:
            context.pos_matrix[:, g][(f_rank >= V_min) & (f_rank < V_max)] = context.net_value_left[g]

    context.pos_matrix[:, context.group_num] = context.net_value_left[context.group_num]

    # 组内等权
    context.pos_matrix = context.pos_matrix / np.count_nonzero(context.pos_matrix, axis=0)

    # 每个资产的仓位=现金比例/股票价格
    for g in range(context.group_num + 1):
        context.pos_matrix[:, g] = context.pos_matrix[:, g] / context.price.loc[context.td, :].values
    # context.xiaoma = pd.DataFrame(context.pos_matrix,index=context.codes)

    tmp = [context.td]
    for g in range(context.group_num):
        context.xiaowei = context.pos_matrix[:, g].nonzero()[0]
        tmp.append(np.array(pd.Series(context.codes).iloc[context.xiaowei].to_list()))
    context.stock_position.append(tmp)
    # 存储换仓数据
    context.history['td'].append(context.td)
    context.history['times'] += 1
    context.history['factor'].append(f.values)
    context.history['price'].append(context.price.loc[context.td, :].values)
    context.history['position'].append(context.pos_matrix)

    # 计算上次换仓IC
    if context.last_td_mark:
        # 初次建仓不计算
        stock_return = (context.history['price'][-1] - context.history['price'][-2]) / context.history['price'][-2]
        factor = context.history['factor'][-2]
        corr = pd.DataFrame([stock_return, factor]).T.corr(method='pearson')
        context.history['IC'].append(corr.loc[0, 1])
        rank_corr = pd.DataFrame([stock_return, factor]).T.corr(method='spearman')
        context.history['Rank_IC'].append(rank_corr.loc[0, 1])


def handle_data(context):
    if not context.last_td_mark:
        # 最初建仓
        context.net_value_left = context.net_value.loc[context.td, :].values
        rebalance(context)
    else:
        # 利用仓位矩阵计算净值
        td_price = context.price.loc[context.td, :].fillna(value=0)
        # print(td_price.dot(context.pos_matrix))
        # a=np.isnan(context.pos_matrix)[:,0]
        context.net_value.loc[context.td, :] = td_price.dot(context.pos_matrix)


        # 更新剩余净值
        context.net_value_left = context.net_value.loc[context.td, :].values

        # rebalance_month = [5, 9, 11]
        rebalance_month = [1,5,8,11]
        # 调仓
        if isinstance(context.freq, int) and context.last_day == context.td:
            # 固定交易日换仓
            rebalance(context)
            context.last_td_mark = 0
        elif context.freq == 'd':
            # 每日换仓
            rebalance(context)
        elif context.freq == 'w' and (context.td.strftime('%W') != context.last_td_mark):
            # 每周换仓
            rebalance(context)
        elif context.freq == 'm':
            # 每月换仓
            if context.td.month != context.last_td_mark:
                context.need_rebalance = 1
            if context.need_rebalance and (context.td.day >= 1):
                context.need_rebalance = 0
                rebalance(context)
        elif context.freq == 'f' and (context.td.month in rebalance_month and context.td.month != context.last_td_mark):
            # 5,9,11的第一个交易日换仓,更新仓位矩阵
            rebalance(context)


def run(context):
    initialize(context)

    for td in context.trade_day:
        context.td = td
        handle_data(context)

        # 更改标记，用于判断是否换仓
        if isinstance(context.freq, int):
            context.last_td_mark += 1

            # 如果last_td_mark为1则为换仓日且为新循环第一天，在这里登记新循环的last_day
            context.last_day = td + np.timedelta64(context.freq - 1,
                                                   'D') if context.last_td_mark == 1 else context.last_day
            # 如果last_day不在交易日中，取最近的一个日期作为last_day 最后一个循环暂未取最后一天为last_day
            if context.last_day not in context.trade_day and context.last_day < context.trade_day[-1]:
                context.last_day = context.trade_day[context.trade_day > context.last_day][0]
            # elif context.last_day>context.trade_day[-1]:
            # context.last_day = context.trade_day[-1] #无论如何最后一天换仓用这句
        elif context.freq == 'w':
            context.last_td_mark = td.strftime('%W')
        else:
            context.last_td_mark = td.month

    summary(context)


def summary(context):
    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    codename = context.codename
    context.pos_info=pd.DataFrame(context.stock_position)
    context.pos_info.to_excel('%s/pos_info_'% (codename,)+context.indu+'.xlsx' )
    factor_name = context.Path_factor.split('/')[-1].split('.')[0]
    industry_name = context.indu
    result = {}

    # 计算多空和超额收益
    summary_df = pd.DataFrame(index=context.net_value.index, columns=['Long', 'Short', 'benchmark', 'LS', 'Excess'])
    summary_df['Long'] = context.net_value['group1']
    summary_df['Short'] = context.net_value['group' + str(context.group_num)]
    summary_df['benchmark'] = context.net_value['benchmark']
    summary_df = summary_df.pct_change()
    summary_df['LS'] = summary_df['Long'] - summary_df['Short']
    summary_df['Excess'] = summary_df['Long'] - summary_df['benchmark']
    summary_df = (summary_df + 1).cumprod()
    summary_df.iloc[0, :] = 1

    with PdfPages(factor_name + '_' + industry_name + '_' + str(context.freq) + '.pdf') as pdf:

        # 分层
        fig = plt.figure(figsize=(10, 6))

        plt.plot(context.net_value)
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.legend(['group' + str(i) for i in range(1, context.group_num + 1)] + ['benchmark'])
        plt.title('分层情况')
        pdf.savefig(fig)
        plt.close()

        # 分组回测
        fig = plt.figure(figsize=(10, 6))
        td_num = len(context.trade_day)
        return_year = context.net_value.iloc[-1, :] ** (252 / td_num) - 1

        context.industry_value_list.loc[context.indu, :] = return_year

        plt.bar(context.net_value.columns, return_year, color=context.group_num * ['cyan'] + ['silver'])
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.title('分组年化收益率')
        pdf.savefig(fig)
        plt.close()

        # IC, Rank_IC
        fig = plt.figure(figsize=(10, 6))
        result['IC_mean'] = np.mean(context.history['IC'])
        result['IC_max'] = np.max(context.history['IC'])
        result['IC_min'] = np.min(context.history['IC'])
        result['IC_std'] = np.std(context.history['IC'], ddof=1)
        result['IC_IR'] = np.mean(context.history['IC']) / np.std(context.history['IC'], ddof=1)
        result['IC_T'] = result['IC_IR'] * np.sqrt(context.history['times'] - 1)

        result['Rank_IC_mean'] = np.mean(context.history['Rank_IC'])
        result['Rank_IC_max'] = np.max(context.history['Rank_IC'])
        result['Rank_IC_min'] = np.min(context.history['Rank_IC'])
        result['Rank_IC_std'] = np.std(context.history['Rank_IC'], ddof=1)
        result['Rank_IC_IR'] = np.mean(context.history['Rank_IC']) / np.std(context.history['Rank_IC'], ddof=1)
        result['Rank_IC_T'] = result['Rank_IC_IR'] * np.sqrt(context.history['times'] - 1)

        factor_industry.form_indu = pd.DataFrame([result])

        title = 'IC: 均值:' + '{:.2%}'.format(result['IC_mean']) + \
                '   最大值:' + '{:.2%}'.format(result['IC_max']) + \
                '   最小值:' + '{:.2%}'.format(result['IC_min']) + \
                '   标准差:' + '{:.2%}'.format(result['IC_std']) + \
                '   IR:' + '{:.2f}'.format(result['IC_IR']) + '\n\n' + \
                'Rank_IC: 均值:' + '{:.2%}'.format(result['Rank_IC_mean']) + \
                '   最大值:' + '{:.2%}'.format(result['Rank_IC_max']) + \
                '   最小值:' + '{:.2%}'.format(result['Rank_IC_min']) + \
                '   标准差:' + '{:.2%}'.format(result['Rank_IC_std']) + \
                '   IR:' + '{:.2f}'.format(result['Rank_IC_IR'])
        x_label = [t.strftime('%Y-%m-%d') for t in context.history['td'][:-1]]
        plt.bar(x_label, context.history['Rank_IC'])
        if isinstance(context.freq, int):
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(252 / (2 * context.freq))))
        elif context.freq == 'd':
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(126))
        elif context.freq == 'w':
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(26))
        elif context.freq == 'm':
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(6))
        elif context.freq == 'f':
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        plt.grid(axis="y")
        plt.xticks(rotation=-45)
        plt.title('IC序列\n\n' + title)
        pdf.savefig(fig)
        plt.close()


###输入：因子的列表
# ff_list = ['IVR_FF3factor_6m']
# ff_list = ['IVR_FF3factor_1m', 'IVR_FF3factor_3m', 'IVR_FF3factor_6m', 'bp', 'exp_wgt_return_1m', 'exp_wgt_return_3m', 'exp_wgt_return_6m', 'mean_10', 'mean_120', 'mean_20', 'mean_60', 'ofcp', 'return_1m', 'return_3m', 'return_6m', 'turn_1m', 'turn_3m', 'turn_6m', 'margin_saletradingamount', 'mrg_long_amt', 'mrg_short_vol']
# ff_list = ['IVR_FF3factor_1m', 'IVR_FF3factor_3m', 'IVR_FF3factor_6m']
ff_list = ['bppercent']
start_date = '20220104'
end_date = '20230412'

# #初始化创建一个文件储存IC值表格 #不用写入writer了！命名为writer
factor_industry = Factor_Industry(ff_list)
# df1 = pd.DataFrame([])
# df1.to_excel("IC_result.xlsx", sheet_name="first_sheet")
# writer = pd.ExcelWriter(r"IC_result.xlsx", mode="a", engine="openpyxl")

for i in range(len(ff_list)):
    # print(i)
    ff = ff_list[i]

    path1 = "%s\%s_%s_%s.csv" % (ff, ff, start_date, end_date)
    path2 = "%s\daylist_%s_%s_%s.csv" % (ff, ff, start_date, end_date)
    path3 = "%s\price_%s_%s_%s.csv" % (ff, ff, start_date, end_date)
    # print(path1,path2,path3)
    context = Context(start_date, end_date, 5, 14, path1, path2, path3)
    context.codename = ff
    context.industry_value_list = pd.DataFrame(index=context.industry_list,
                                               columns=['group' + str(i) for i in range(1, context.group_num + 1)] + [
                                                   'benchmark'])
    factor_industry.form = pd.DataFrame(index=context.industry_list,
                                        columns=['IC_mean', 'IC_max', 'IC_min', 'IC_IR', 'IC_T', 'Rank_IC_mean',
                                                 'Rank_IC_max', 'Rank_IC_min', 'Rank_IC_std', 'Rank_IC_IR',
                                                 'Rank_IC_T'])

    ##industry_list为行业信息 按行业读取信息
    for j in context.industry_list:

        # print(j)
        # print(j)
        ##context.indu为行业名称 中文形式
        context.indu = j
        ##生成代码
        codes_industry = list(
            context.industry[context.industry["industry"] == j]["code"])  ###codes_industry是该行业中的所有股票代码
        context.price = pd.read_csv(context.Path_price, index_col=0, parse_dates=[0])  # parse_date参数：解析第0列为日期格式
        context.factor = pd.read_csv(context.Path_factor, index_col=0, parse_dates=[0])
        codes_all = list(context.price.columns)  ###codes_all是数据清洗之后该文件夹内的所有股票代码
        codes_c = list(context.factor.columns)

        context.codes = list(set(codes_all) & set(codes_industry) & set(codes_c))

        # print(len(context.codes))
        if len(context.codes) < 7:
            continue

        run(context)

        factor_industry.form.loc[j, :] = factor_industry.form_indu.iloc[0, :]

    ###写入excel
    # factor_industry.form.to_excel(writer, sheet_name="%s" % ff)
    # print(factor_industry.form)
    print("-----------------------------", ff, "检验结果————————————————————————————————")
    print("IC均值为：", factor_industry.form.loc[:, "IC_mean"].mean(skipna=True))
    print("IC_IR均值为：", factor_industry.form.loc[:, "IC_IR"].mean(skipna=True))
    print("Rank_IC均值为：", factor_industry.form.loc[:, "Rank_IC_mean"].mean(skipna=True))
    print("Rank_IC_IR均值为：", factor_industry.form.loc[:, "Rank_IC_IR"].mean(skipna=True))

    factor_industry.form.to_excel("%s_%s%s.xlsx" % (ff, start_date, end_date))

    setattr(factor_industry, ff, context.industry_value_list)
    factor_industry.total.loc[ff, :] = context.industry_value_list.mean(axis=0, skipna=True)
    context.industry_value_list.to_excel("%s\%s.xlsx" % (ff, ff), sheet_name="%s" % (ff))

# writer.save()
# writer.close()
