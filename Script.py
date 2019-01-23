#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEMS 308 - HW1

"""

# imprting libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#importing dataset
mylist = []

for chunk in  pd.read_csv('Medicare.txt', sep='\t',chunksize=20000):
    mylist.append(chunk)

data = pd.concat(mylist, axis= 0)
data = data.iloc[1:]
del mylist
del chunk

#subsetting dataset
data = data[data.MEDICARE_PARTICIPATION_INDICATOR == 'Y']
data = data[data.NPPES_PROVIDER_STATE == 'FL']

# want only Type 1 CPT AMA physican care (disregard type 2, which are medical supplies) of procedure type surgery
data['IS_CPT'] = data.HCPCS_CODE.str.isdigit()
data = data[data.IS_CPT == True]
data['CPT_CODE'] = data.HCPCS_CODE.str[:2].astype('str').astype(int)
data['SURGERY'] = (10 <= data.CPT_CODE) & (data.CPT_CODE <= 69)
data = data[data.SURGERY == True]

#filling NA
data.NPPES_CREDENTIALS.fillna('ORG',inplace = True)
data.NPPES_PROVIDER_GENDER.fillna('U',inplace = True)

data_untransformed = data

# One Hot Encoding
cat_features = ['NPPES_CREDENTIALS','NPPES_PROVIDER_GENDER','PROVIDER_TYPE','PLACE_OF_SERVICE','CPT_CODE']

for cat in cat_features:
    data = pd.get_dummies(data, columns = [cat])

# getting full features column names 
features_names = ['LINE_SRVC_CNT','BENE_UNIQUE_CNT','BENE_DAY_SRVC_CNT','AVERAGE_MEDICARE_ALLOWED_AMT' \
                  ,'NPPES_CREDENTIALS','NPPES_PROVIDER_GENDER',\
                  'PROVIDER_TYPE','PLACE_OF_SERVICE','CPT_CODE']

features = []

for ft in features_names:
    filter_col = [col for col in data if col.startswith(ft)]
    for col in filter_col:
        features.append(col)


#K-Mean clustering

feature_data = data.loc[:,features].values
scaler = StandardScaler()
scaler.fit(feature_data) 
feature_data = scaler.transform(feature_data)                   

score = []

for k in range(2,9):
    kmeans = KMeans(n_clusters=k, max_iter = 3000)
    kmeans.fit(feature_data) 
    score.append(kmeans.inertia_)

plt.plot(range(2,9), score, '-o')
plt.xlabel('k')
plt.ylabel('score')

# selected number of clusters
k = 5
min_score = 9999999999999999

# after selecting cluster, run 10 times to get best result
for i in range(1,11):
    kmeans = KMeans(n_clusters=k,init = 'random' ,max_iter = 3000)
    kmeans.fit(feature_data)
    if (kmeans.inertia_ < min_score):
        best_kmean = kmeans
        min_score = kmeans.inertia_
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

pca = PCA(n_components=2).fit(feature_data.data)
data_2d = pca.transform(feature_data.data)

pca = PCA(n_components=2).fit(centers.data)
cluster_2d = pca.transform(centers.data)

plt.scatter(data_2d[:,0], data_2d[:,1], c=labels, cmap='rainbow')
plt.ylim(-5,5)

#analysis of results
data_untransformed['cluster'] = labels

data_untransformed['total_submitted_amt'] = data_untransformed['LINE_SRVC_CNT'] * data_untransformed['AVERAGE_SUBMITTED_CHRG_AMT']       
data_untransformed['total_payment_amt'] = data_untransformed['LINE_SRVC_CNT'] * data_untransformed['AVERAGE_MEDICARE_PAYMENT_AMT'] 


grouped = data_untransformed[['LINE_SRVC_CNT','total_submitted_amt','total_payment_amt']].groupby(data_untransformed['cluster']) 
     
results = grouped.sum()

grouped = data_untransformed['cluster'].groupby(data_untransformed['cluster'])  
counts = grouped.count()

grouped = data_untransformed['AVERAGE_MEDICARE_ALLOWED_AMT'].groupby(data_untransformed['cluster'])  
allowed = grouped.mean()

results["cluster"] = counts
results['allowed'] = allowed
results['avg_payment'] = results['total_payment_amt'] / results['LINE_SRVC_CNT']
results['avg_submit'] = results['total_submitted_amt'] / results['LINE_SRVC_CNT']
results['diff_submit_pay'] = results['avg_submit'] - results['avg_payment']
results['diff_allowed_submit'] = results['allowed'] - results['avg_payment']

total_pay = results['total_payment_amt'].sum() 

results['fraction_of_payment'] = results['total_payment_amt'] / total_pay * 100
       
#describing cluster by CPT type
grouped = data_untransformed['CPT_CODE'].groupby(data_untransformed['cluster'])    
grouped.median()






