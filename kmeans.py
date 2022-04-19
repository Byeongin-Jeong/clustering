# library import
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import csv
import random

from pandas import merge
from scipy import stats
from collections import Counter
from plotly.offline import init_notebook_mode, iplot
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

init_notebook_mode(connected=True)

# 2D 그래프로 표현
def gettrace(patient_clusters, cluster_num):
    r = random.randrange(1,255)
    g = random.randrange(1,255)
    b = random.randrange(1,255)
    return go.Scatter(x = patient_clusters[patient_clusters.cluster == cluster_num]['x'],
                   y = patient_clusters[patient_clusters.cluster == cluster_num]['y'],
                   name = "Cluster {}".format(cluster_num+1),
                   mode = "markers",
                   marker = dict(size = 10,
                                 color = "rgba({0}, {1}, {2}, 0.5)".format(r, g, b),
                                 line = dict(width = 1, color = "rgb(0,0,0)")))

# 최적의 클러스터 개수를 추출하기 위한 함수
def elbow(X):
    sse = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i, init='k-means++', random_state=0)
        km.fit(X)
        sse.append(km.inertia_)
        print (km.inertia_)
    plt.plot(range(1, 10), sse, marker='o')
    plt.xlabel('cluster count')
    plt.ylabel('SSE')
    plt.show()

src_dic = {}
dst_dic = {}

loopcnt = 0

f = open('data/netflow_dump.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

for line in rdr:
    src = line[0]
    dst = line[1]

    src_dic.setdefault(src, set())
    src_dic[src].add(dst)
f.close()

dst_li = [len(x) for x in src_dic.values()]

# 고유한 DST ADDR 수집
id_dst_dic = {}
for dst_set_li in src_dic.values():
    for x in dst_set_li:
        if x not in id_dst_dic:
            id_dst_dic.setdefault(x, len(id_dst_dic))

print("# of left dst:%d" % (len(id_dst_dic)))

id_src_dic = {}

# DST ADDR 개수
all_dst_count = len(id_dst_dic)

produc_columns = [x for x in id_dst_dic.values()]
produc_columns.sort()
produc_columns.insert(0, 'srcip')

df = pd.DataFrame(columns=produc_columns)

for srcip, dsts in src_dic.items():
    src_dst_vec = [0] * all_dst_count

    id_src_dic[len(id_src_dic)] = srcip

    for dstip in dsts:
        src_dst_vec[id_dst_dic[dstip]] = 1

    # 첫번째 columns를 src ip로한 dataframe 생성
    src_dst_vec.insert(0, srcip)
    df.loc[len(id_src_dic)] = [n for n in src_dst_vec]

# 데이터를 랜덤으로 섞는다    
df = shuffle(df)

# 데이터의 70%만 추출
tran_lenght = int(len(df) * 0.7)

# 학습 데이터와 테스트 데이터로 분리
train_data = df[df.columns[1:]][:tran_lenght+1]
test_data = df[df.columns[1:]][tran_lenght:]

print ("70% data count: ", tran_lenght)

# 최적의 클러스터 확인
elbow(train_data)

cols = df.columns[1:]

clusternum = 4

# 데이터 학습
kmeans = KMeans(n_clusters = clusternum)
kmeans.fit(train_data)

# 테스트 데이터 클러스터
test_clusters = df[:][tran_lenght:]
test_clusters["cluster"] = kmeans.predict(test_data)
test_clusters.tail()

# 좌표 데이터 추출
pca = PCA(n_components = 2)
test_clusters['x'] = pca.fit_transform(test_clusters[cols])[:, 0]
test_clusters['y'] = pca.fit_transform(test_clusters[cols])[:, 1]
test_clusters.tail()

# dataframe 재정의(ip, cluster, x, y)
patient_clusters = test_clusters[['srcip', 'cluster', 'x', 'y']]
patient_clusters.tail()

# 2D 그래프로 시각화
data = []
for idx in range(clusternum):
    data.append(gettrace(patient_clusters, idx))

iplot(data)
