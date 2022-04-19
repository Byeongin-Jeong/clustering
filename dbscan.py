# library import
import pandas as pd
import mglearn
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import random
init_notebook_mode(connected=True)

# Importing the dataset
df = pd.read_csv('data/DBSCAN_data_2ea.csv')
df.tail()

# 차원 데이터를 바탕으로 좌표 데이터 추출(x, y, z)
cols = df.columns[1:]
pca = PCA(n_components = 3)
df['x'] = pca.fit_transform(df[cols])[:, 0]
df['y'] = pca.fit_transform(df[cols])[:, 1]
df['z'] = pca.fit_transform(df[cols])[:, 2]
df.tail()

# 좌표 데이터 스케일 축소(좌표 범위가 크기 때문)
def conv(i):
    return [int(df[i:i+1]['x']), int(df[i:i+1]['y']), int(df[i:i+1]['z'])]

convlist = []
for ldx in range(0, len(df)):
    convb = conv(ldx)
    convlist.append(convb)

rawpoint = StandardScaler().fit_transform(convlist)

# Data Traning (**esp**(기준점부터의 거리) = 0.2, **min_samples**(반경내 있는 점의수) = 20)
# 학습과 동시에 클러스터링 결과가 출력된다.
db = DBSCAN(eps=0.2, min_samples=20)
y_pred = db.fit_predict(rawpoint)
df["cluster"] = y_pred

clusternum = max(y_pred)

print ("clusterNum: {}".format(clusternum))
df.head()

# 2D 그래프 출력
mglearn.discrete_scatter(rawpoint[:, 0], rawpoint[:, 1], y_pred, markers='o')

# 클러스터링 제외된 데이터 ID 추출
df.loc[df['cluster'] == -1].head()

# 3D 출력 함수 (클러스터링이 되지 않은 Noise point는 빨간색으로 표시)
def gettrace(cl_result, cluster_num):
    if cluster_num == -1:
        r = 255
        g = 0
        b = 0
    else:
        r = random.randrange(50,255)
        g = random.randrange(50,255)
        b = random.randrange(50,255)

    return go.Scatter3d(x = [x[0] for x in cl_result[cluster_num]],
                        y = [y[1] for y in cl_result[cluster_num]],
                        z = [z[2] for z in cl_result[cluster_num]],
                        mode='markers',
                        marker=dict(
                            color="rgba({0}, {1}, {2}, 0.5)".format(r, g, b),
                            size=5,
                            symbol='circle',
                            line = dict(width = 1, color = "rgb(0,0,0)"),
                        opacity=0.9
                        ))

# 3D 그래프로 출력 가능하도록 데이터 가공
cluster_result = {}
for i, c in enumerate(y_pred):
    cluster_result.setdefault(c, [])
    cluster_result[c].append(rawpoint[i])

# 3D 데이터 출력
data = []
for idx in range(-1, clusternum +1):
    if idx in cluster_result:
        data.append(gettrace(cluster_result, idx))

iplot(data)
