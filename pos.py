import pandas as pd
#%%
industry = pd.read_csv("industry申万三级.csv", encoding='gbk')
code = '603982.SH'
factor = 'bppercent'

cengshu = 1

#%%
ind = industry[industry['code']==code].iloc[0,1]
aa = pd.read_excel('%s/pos_info_'%(factor,)+ind+'.xlsx',index_col = 1,parse_dates = [0])
aa.drop('Unnamed: 0',axis=1,inplace = True)

df = pd.DataFrame(index=aa.index,columns = ['查找的票所在的层数'])

for i in aa.index:
    b = aa.loc[i,:].str.find(code, start=0, end=None)
    df.loc[i,:] = list(b[b>-1].index)[0]

print(df)

#%%
duiying = pd.read_excel('全部A股.xlsx')
print(duiying[duiying['证券代码'] == '000002.SZ'].iloc[0,0])

#%%
def codes_to_names(codes,duiying):
    codes = codes[2:-2]
    codes = codes.replace('\n', '')
    codes_list = codes.split("' '")

    names_list = []
    for j in codes_list:
        name = duiying[duiying['证券代码'] == j].iloc[0, 1]
        names_list.append(name)
    str = ','
    names = str.join(names_list)
    return codes,names


for i in aa.index:
    codes = aa.loc[i,cengshu]
    print(codes)
    codes,names = codes_to_names(codes,duiying)

    df.loc[i,'最差一层'] = codes
    df.loc[i,'最差一层名称'] = names

df.to_excel('%s/pos_info_%s'%(factor,code)+ind+'.xlsx')
