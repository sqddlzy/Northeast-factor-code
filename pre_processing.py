##用于批量处理因子值和价格值
import pandas as pd
from datetime import datetime
import numpy as np

# 参数
ff_list = ['bppercent']
start_date = '20220104'
end_date = '20230412'

for i in range(len(ff_list)):
    factor_name = ff_list[i]

    factor_frame_raw = pd.read_excel('%s\%s_%s_%s.xlsx' % (factor_name, factor_name, start_date, end_date), index_col=0)
    # price_frame_raw = pd.read_csv('%s\price_%s_%s_%s.csv' % (factor_name, factor_name, start_date, end_date),
    #                                 index_col=0)
    price_frame_raw = pd.read_csv('close.csv',index_col=0)
    # for column in factor_frame_raw.columns:
    #     factor_frame_raw[column] = pd.to_numeric(factor_frame_raw[column])
    # for column in price_frame_raw.columns:
    #     price_frame_raw[column] = pd.to_numeric(price_frame_raw[column])
    price_frame_raw.info()
    if type(factor_frame_raw.index[0]) == str:
        factor_frame_raw.index = factor_frame_raw.index.map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    if type(price_frame_raw.index[0]) == str:
        price_frame_raw.index = price_frame_raw.index.map(lambda x: datetime.strptime(x, '%Y/%m/%d'))
    # factor_frame_raw.index = factor_frame_raw.index.map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # price_frame_raw.index = price_frame_raw.index.map(lambda x: datetime.strptime(x, '%Y/%m/%d'))
    print(price_frame_raw)
    # 删掉有0的列
    # tmp = factor_frame_raw[["688278.SH", "689009.SH"]]
    # ymppppp=tmp.apply(lambda x: (x != 0).all())
    # factor_frame_clear_zero = factor_frame_raw.loc[:, factor_frame_raw.apply(lambda x: (x != 0).all())]
    # factor_frame_clear_zero = factor_frame_raw.loc[~(factor_frame_raw == 0).all(axis=1)]
    # # 删掉有0的行
    # factor_frame_spicking = factor_frame_clear_zero.loc[:, ~(factor_frame_clear_zero == 0).all(axis=0)]
    # 删掉有nan的列
    factor_frame = factor_frame_raw.dropna(axis=1, how='any')

    # 生成daylist表格
    daylist_frame = pd.DataFrame(index=factor_frame.index)
    # daylist_frame.index.set_names = ["datetime"]
    # print(daylist_frame)
    # 生成price表格
    price_frame = price_frame_raw.loc[factor_frame.index, factor_frame.columns]
    price_frame = price_frame.loc[:, price_frame.apply(lambda x: (x != 0).all())]
    factor_frame = factor_frame.loc[price_frame.index, price_frame.columns]

    factor_frame.to_csv(
        "%(factor_name)s\%(factor_name)s_%(start)s_%(end)s.csv" % {"factor_name": factor_name, "start": start_date,
                                                                   "end": end_date})
    daylist_frame.to_csv("%(factor_name)s\daylist_%(factor_name)s_%(start)s_%(end)s.csv" % {"factor_name": factor_name,
                                                                                            "start": start_date,
                                                                                            "end": end_date})
    price_frame.to_csv("%(factor_name)s\price_%(factor_name)s_%(start)s_%(end)s.csv" % {"factor_name": factor_name,
                                                                                        "start": start_date,
                                                                                        "end": end_date})
