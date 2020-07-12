# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:37:32 2020

@author: 86178
"""

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

sheet_name=['2015','2016','2017','2018','会员等级']
sheet_datas=[pd.read_excel('sales.xlsx',sheet_name=i) for i in sheet_name]

#数据审查
for each_name , each_data in zip(sheet_name, sheet_datas):
    print('[data summary for {0:=^50}]'.format(each_name))
    print('Overview:','\n',each_data.head(4))
    print('DESC:','\n',each_data.describe())
    print('NA records',each_data.isnull().any(axis=1).sum())
    print('Dtypes',each_data.dtypes)
#数据预处理
for ind,each_data in enumerate(sheet_datas[:-1]):
    sheet_datas[ind]=each_data.dropna()
    sheet_datas[ind]=each_data[each_data['订单金额']>1]
    sheet_datas[ind]['max_year_date']=each_data['提交日期'].max()    
    
data_merge=pd.concat(sheet_datas[:-1],axis=0)
data_merge['date_interval']=data_merge['max_year_date']-data_merge['提交日期']
data_merge['year']=data_merge['提交日期'].dt.year
#转化日期
data_merge['date_interval']=data_merge['date_interval'].apply(lambda x:x.days)
rfm_gb=data_merge.groupby(['year','会员ID'],as_index=False).agg({'date_interval':'min','提交日期':'count','订单金额':'sum'})
rfm_gb.columns=['year','会员ID','r','f','m']

desc_pd=rfm_gb.iloc[:,2:].describe().T
print(desc_pd)
#分箱
r_bins=[-1,79,255,365]
f_bins=[0,2,5,130]
m_bins=[0,69,1199,206252]

rfm_merge=pd.merge(rfm_gb,sheet_datas[-1],on='会员ID',how='inner')

clf=RandomForestClassifier()
clf=clf.fit(rfm_merge[['r','f','m']],rfm_merge['会员等级'])
weights=clf.feature_importances_

rfm_gb['r_score']=pd.cut(rfm_gb['r'],r_bins,labels=[i for i in range(len(r_bins)-1,0,-1)])
rfm_gb['f_score']=pd.cut(rfm_gb['f'],f_bins,labels=[i+1 for i in range(len(f_bins)-1)])
rfm_gb['m_score']=pd.cut(rfm_gb['m'],m_bins,labels=[i+1 for i in range(len(m_bins)-1)]) 


rfm_gb=rfm_gb.apply(np.int32)
rfm_gb['rfm_score']=rfm_gb['r_score']*weights[0]+rfm_gb['f_score']*weights[1]+rfm_gb['m_score']*weights[2]

#保存
rfm_gb.to_excel('sales_rfm_score.xlsx')































































































