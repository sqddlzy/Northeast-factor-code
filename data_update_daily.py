from iFinDPy import *
import iFinDPy as fin
import numpy as np
import pandas as pd

daylist = pd.read_csv("daylist.csv", header=None, encoding='utf-8', engine='python')
stock = pd.read_csv("stock_info.csv", header=None, encoding='gbk', engine='python')

fin.THS_iFinDLogin('dbzq2122', 'abcd1234')

# 0:successfully log in
# -201: repeatly log in
def tradedays_list(start, end):
    '''
    计算两个日期间的工作日
    start:开始时间
    end:结束时间
    '''
    from datetime import datetime, timedelta
    from chinese_calendar import is_holiday
    # 字符串格式日期的处理
    if type(start) == str:
        start = datetime.strptime(start, '%Y-%m-%d').date()
    if type(end) == str:
        end = datetime.strptime(end, '%Y-%m-%d').date()
    # 开始日期大，颠倒开始日期和结束日期
    if start > end:
        start, end = end, start

    list = []
    while True:
        if start > end:
            break
        if is_holiday(start) or start.weekday() == 5 or start.weekday() == 6:
            start += timedelta(days=1)
            continue
        temp = start.strftime('%Y-%m-%d')
        list.append(temp)
        start += timedelta(days=1)
    return list

from datetime import datetime
from datetime import timedelta

####获得list形式的时间戳序列###
temp = len(daylist)
day = []
for i in range(0, temp):
    day.append(daylist.iloc[i][0])
# type of day is list, data in it is timestamp.

###get now in the formation of '%Y-%m-%d'###
now_time = datetime.now()
today = now_time.strftime('%Y-%m-%d')  # datatime 格式转化为str
# today str

###get yesterday in the formation of '%Y-%m-%d'###
yesterday_time = now_time - timedelta(days=1)
yesterday = yesterday_time.strftime('%Y-%m-%d')  # datatime 格式转化为str
# yesterday str

###get latest timestamp in the formation of '%Y-%m-%d'###
timestamp = day[-1]
date_time = datetime.strptime(timestamp, '%Y-%m-%d').date()
date_time += timedelta(days=1)
# date_time = timestamp.to_pydatetime() #trans Timestamp formation into datetime formation
latest = date_time.strftime("%Y-%m-%d")
# latest str

###get begin timestamp in the formation of '%Y-%m-%d'###
timestampbegin = day[0]
date_timebegin = datetime.strptime(timestampbegin, '%Y/%m/%d').date()
# date_time = timestamp.to_pydatetime() #trans Timestamp formation into datetime formation
begin = date_timebegin.strftime("%Y-%m-%d")
# begin str

###test usage###
# date_time = datetime.fromtimestamp(timestamp)
# print("date and time:", date_time.strftime("%Y-%m-%d"))
# type(latest)

###获得需要更新的股票序列###
# stock_list是一个字符串，Stock_list是一个列表
stock_amount = len(stock)
Stock_list = []
for i in range(0, stock_amount):
    Stock_list.append(stock[0][i])
stock_list = ','.join(Stock_list)


# 更新的csv文件的名称
name_list = \
    ['amount', 'close_hfq', 'floatshare', 'high_hfq', 'low_hfq', 'open_hfq', 'totalShares', 'volume',
     'cap', 'transactionAmount', 'turnover', 'PE_ttm', 'vwap','ps_ttm','change_ratio','pb_lf']
# 参数
parameter_list = \
    ['amount', 'close', 'floatSharesOfAShares', 'high', 'low', 'open', 'totalShares', 'volume',
     'totalCapital', 'transactionAmount', 'turnoverRatio', 'pe_ttm', 'avgPrice','ps','changeRatio','pb']
# 之后往后面添加

# 历史行情-后复权（分红再投）-数据缺省-通用模块
def download_hfq(stock_list, Stock_list, start, end, name_list, parameter_list):
    ###循环进行更新文件的数据###
    for i in range(0, len(name_list)):
        o = pd.DataFrame()
        data_array = []
        count = len(Stock_list)  #股票数量
        name = name_list[i]
        parameter = parameter_list[i]

        # 获得更新两天所差的时间
        from datetime import datetime, timedelta
        from chinese_calendar import is_holiday
        # 字符串格式日期的处理
        if type(start) == str:
            starting = datetime.strptime(start, '%Y-%m-%d').date()
        if type(end) == str:
            ending = datetime.strptime(end, '%Y-%m-%d').date()
        # 开始日期大，颠倒开始日期和结束日期
        if starting > ending:
            starting, ending = ending, starting
        counts = 0
        while True:
            if starting > ending:
                break
            if is_holiday(starting) or starting.weekday() == 5 or starting.weekday() == 6:
                starting += timedelta(days=1)
                continue
            counts += 1
            starting += timedelta(days=1)  #datetime格式的加减需要用timedelta这个函数
        delay = counts   #delay:返回start和end之间间隔交易日天数

        update_list = tradedays_list(start, end)

        # 下载数据，store是核心函数，CPS：3，fill Omit是后复权分红再投，以后写别的模块的时候注意改这个函数
        store = THS_HQ(stock_list, parameter, 'CPS:3,fill:Omit', start, end, 'format:dataframe')  # 下载全时间全股票数据
        # 此时store的数据格式并不是dataframe，后面需要格式转化
        temp = store.data
        # for这一段是核心代码
        for i in range(0, count):
            code = Stock_list[i]  # 依次选择股票
            origin = temp.loc[temp["thscode"] == code]  # 按照次序分股票进行切片
            flag = len(origin)  # 判断是否退市，退市为0
            if flag == 0:
                x_list = [0] * delay   #若退市全部赋值0
            #                 list_nan = np.full([1,delay], np.nan)
            #                 x_list = list_nan[0].tolist() #用np.nan填充所有空缺数据
            else:  #若不退市，先赋值0，再把数据填充
                x_list = [0] * delay
                #                 list_nan = np.full([1,delay], np.nan)
                #                 x_list = list_nan[0].tolist() #先用np.nan填充所有数据
                for i in range(0, len(origin)):
                    # 调取index = i行的时间戳以及相应数据
                    time = origin.iloc[i][0]
                    o = origin.iloc[i][2]
                    # 匹配上时间戳则将相应索引的np.nan覆写掉，若没匹配上则位置上仍是np.nan
                    index = update_list.index(time)
                    x_list[index] = o
            # 股票合并，data-array是所有股票数据，其中退市和停盘都以0填充
            data_array.append(x_list)  # open_array的横轴是时间戳，竖轴是股票代码（需转置）
        o = pd.DataFrame(data_array)
        o = o.T  # 转置到原矩阵行列格式
        # 写入行列标签，其中行标签为时间戳，列标签为股票代码
        s1 = pd.Series(Stock_list)
        s2 = pd.Series(update_list)
        o.columns = s1
        o.index = s2

        # 如果今天数据尚未上传至数据库，则删除今天数据并且更改返回的状态

        notyet = 0
        keyword = len(o.loc[(o == 0).all(axis=1)])
        if keyword != 0:
            notyet = 1
            o = o.T
            o = o.loc[:, (o != 0).any(axis=0)]  # 原函数只能做到删除列，故此进行两次转置
            o = o.T
        # 获得此dataframe直接用于运算则选择以下部分作为输出

        # 获得此dataframe直接用于储存则选择以下部分作为输出
        # 写入csv，完成下载,其中保存行列名称，时间字符串存储格式为日期
        o.to_csv('%s.csv' % (name), index=True, header=False, mode='a+', date_format=[0])
    #         o.to_csv('C:/Users/86180/Desktop/update/%s.csv'%(name), mode='a+', index=True,  header=False, date_format=[0])
    return notyet
# 0代表有今天数据，1代表没有今天数据

#这一段最后跑！因为他们是共用一个daylist
# 原有指标数据更新
flag = download_hfq(stock_list, Stock_list, latest, today, name_list, parameter_list)
###更新完数据后，更新时间戳并保存###
if flag == 0:
    update_list = tradedays_list(latest, today)
else:
    update_list = tradedays_list(latest, yesterday)
update_df = pd.DataFrame(update_list)
update_df.to_csv('daylist.csv', mode='a+', date_format=[0], index=False, header=False)
print(flag)  # 打印更新状态，0为今天数据已下载，1为今天数据未下载。
print(update_list[-1])  # 验证时间序列更新的最后一天是否与更新状态对应


#########################################################################################################################

# 下载用
# 日期序列-通用模块（还有bug）
def download_time(stock_list, Stock_list, start, end, name_list, parameter_list):
    ###循环进行更新文件的数据###
    for i in range(0, len(name_list)):
        o = pd.DataFrame()
        data_array = []
        count = len(Stock_list)
        name = name_list[i]
        parameter = parameter_list[i]

        # 获得更新两天所差的时间
        from datetime import datetime, timedelta
        from chinese_calendar import is_holiday
        # 字符串格式日期的处理
        if type(start) == str:
            starting = datetime.strptime(start, '%Y-%m-%d').date()
        if type(end) == str:
            ending = datetime.strptime(end, '%Y-%m-%d').date()
        # 开始日期大，颠倒开始日期和结束日期
        if starting > ending:
            starting, ending = ending, starting
        counts = 0
        while True:
            if starting > ending:
                break
            if is_holiday(starting) or starting.weekday() == 5 or starting.weekday() == 6:
                starting += timedelta(days=1)
                continue
            counts += 1
            starting += timedelta(days=1)
        delay = counts

        update_list = tradedays_list(start, end)

        # 下载数据
        # 按照60天一组进行分组#
        a = update_list
        length = len(a)
        count = 0  # 初始化天数计数
        knot = 0  # 初始化节数计数
        # 预先请求空间
        start_list = []
        end_list = []
        while (count <= length):
            start_list.append(a[count])  # 每个时间节的开头时刻的列表
            count += 59  # 时间节长度为60
            # 如果向后移动60格后还在序列内，则设定其为时间节结尾时刻，如果不在，则设定序列最后一个为其结尾时刻
            if (count <= length):
                end_list.append(a[count])  # 每个时间节的结尾时刻的列表
            else:
                end_list.append(a[-1])
            count += 1
            knot += 1  # 时间节计数

        temp = pd.DataFrame()
        ###由于“store”行请求服务器端口存在单次数据量限制，因此当从开始时刻下载所有数据时，会出现数据过量并且不返回数据的情况###
        ###因此对于长时间段的更新进行时间分段，再按照每个时间节进行下载，下载后拼合在一起以得到完整数据###
        for i in range(0, knot):
            dataknot = THS_HQ(stock_list, parameter, '100', 'Fill:Blank', start_list[i], end_list[i])  # 下载全时间全股票数据
            dataknot_df = pd.DataFrame(dataknot.data)
            con_list = [temp, dataknot_df]
            temp = pd.concat(con_list, axis=0)

        for i in range(0, len(Stock_list)):
            code = Stock_list[i]  # 依次选择股票
            origin = temp.loc[temp["thscode"] == code]  # 按照次序分股票进行切片
            flag = len(origin)  # 判断是否退市，退市为0
            if flag == 0:
                x_list = [0] * delay
            #                 list_nan = np.full([1,delay], np.nan)
            #                 x_list = list_nan[0].tolist() #用np.nan填充所有空缺数据
            else:
                x_list = [0] * delay
                #                 list_nan = np.full([1,delay], np.nan)
                #                 x_list = list_nan[0].tolist() #先用np.nan填充所有数据
                for i in range(0, len(origin)):
                    # 调取index = i行的时间戳以及相应数据
                    time = origin.iloc[i][0]
                    o = origin.iloc[i][2]
                    # 匹配上时间戳则将相应索引的np.nan覆写掉，若没匹配上则位置上仍是np.nan
                    index = update_list.index(time)
                    x_list[index] = o
            data_array.append(x_list)  # open_array的横轴是时间戳，竖轴是股票代码（需转置）
        o = pd.DataFrame(data_array)
        o = o.T  # 转置到原矩阵行列格式
        # 写入行列标签，其中行标签为时间戳，列标签为股票代码
        s1 = pd.Series(Stock_list)
        s2 = pd.Series(update_list)
        o.columns = s1
        o.index = s2
        # 获得此dataframe直接用于运算则选择以下部分作为输出

        # 获得此dataframe直接用于储存则选择以下部分作为输出
        # 写入csv，完成下载,其中保存行列名称，时间字符串存储格式为日期
        #         o.to_csv('%s.csv'%(name), index=True,  header=True, date_format = [0])
        o.to_csv('%s.csv' % (name), mode='a+', index=True, header=True, date_format=[0])
    return 0

########################################################################################################################
new_name_list = ['dividend_rate']
new_parameter_list = ['ths_dividend_rate_stock_stock']

########################################################################################################################
testday = '2023-02-01'
########################################################################################################################
# 新增指标数据下载
download_time(stock_list, Stock_list, testday, today, new_name_list, new_parameter_list)
########################################################################################################################
# 添加新板块
# 命名方式参照：板块名-调节参数1-调节参数2-……-调节参数N-模板类型
# def download(Stock_list, start, end, name_list, parameter_list):
#     ###循环进行更新文件的数据###
#     for i in range(0, len(name_list)):
#         o = pd.DataFrame()
#         data_array = []
#         count = len(Stock_list)
#         name = name_list[i]
#         parameter = parameter_list[i]

#         # 获得更新两天所差的时间
#         from datetime import datetime,timedelta
#         from chinese_calendar import is_holiday
#             # 字符串格式日期的处理
#         if type(start) == str:
#             starting = datetime.strptime(start,'%Y-%m-%d').date()
#         if type(end) == str:
#             ending = datetime.strptime(end,'%Y-%m-%d').date()
#         # 开始日期大，颠倒开始日期和结束日期
#         if starting > ending:
#             starting,ending = ending,starting
#         counts = 0
#         while True:
#             if starting > ending:
#                 break
#             if is_holiday(starting) or starting.weekday()==5 or starting.weekday()==6:
#                 starting += timedelta(days=1)
#                 continue
#             counts += 1
#             starting += timedelta(days=1)
#         delay = counts

#         update_list = tradedays_list(start, end)
#         #下载数据
#         for i in range(0, count):
#             code = Stock_list[i]
####################################框出的代码是获取数据的接口，可替换########################################
#             store = THS_HQ(code,parameter,'CPS:3,fill:Omit',start ,end ,'format:dataframe')
##############################################################################################################
#             origin = pd.DataFrame(store.data)
#             flag = len(origin)#判断是否退市，退市为0
#             if flag == 0:
#                 x_list = [0]*delay
# #                 list_nan = np.full([1,delay], np.nan)
# #                 x_list = list_nan[0].tolist() #用np.nan填充所有空缺数据
#             else:
#                 x_list = [0]*delay
# #                 list_nan = np.full([1,delay], np.nan)
# #                 x_list = list_nan[0].tolist() #先用np.nan填充所有数据
#                 for i in range(0, len(origin)):
#                     #调取index = i行的时间戳以及相应数据
#                     time = origin.iloc[i][0]
#                     o = origin.iloc[i][2]
#                     #匹配上时间戳则将相应索引的np.nan覆写掉，若没匹配上则位置上仍是np.nan
#                     index = update_list.index(time)
#                     x_list[index] = o
#             data_array.append(x_list)#open_array的横轴是时间戳，竖轴是股票代码（需转置）
#         o = pd.DataFrame(data_array)
#         o = o.T#转置到原矩阵行列格式
#         #写入行列标签，其中行标签为时间戳，列标签为股票代码
#         s1 = pd.Series(Stock_list)
#         s2 = pd.Series(update_list)
#         o.columns = s1
#         o.index = s2
#         #获得此dataframe直接用于运算则选择以下部分作为输出


#         #获得此dataframe直接用于储存则选择以下部分作为输出
#             #写入csv，完成下载,其中保存行列名称，时间字符串存储格式为日期
# #         o.to_csv('%s.csv'%(name), index=True,  header=True, date_format = [0])
#         o.to_csv('%s.csv'%(name), mode='a+', index=False,  header=False, date_format=[0])
#     return 0


#%%

