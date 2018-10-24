import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkContext

data = pd.read_csv("../Tesserae/iris.data", header = None)
data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
data.describe()


usecols = ['sepal_length',
                    'sepal_width',
                    'petal_length',
                    'petal_width',
                    'class'])

data.describe()
data2 = data.
data2.head(5)

data = pd.read_csv("../Tesserae/derived_features_9_23.csv")
data.describe()
data.shape
data1 = data.iloc[:, [0,5,9,10]]
data1.describe()
data2 = data.sample(n = 100).iloc[:,[0,5,9,10]]
data2.reset_index()

data2.rename(columns={'id': 'user_id', 'acute'})
data2.filter(regex = '^acute.*time$').head(5).describe()
data2.iloc[:,0:3].head(5).describe()
data2['id'].nuinque()
data2.describe()
data2.describe()
data3 = data.dropna()
print(data3)
data.dtypes
data['snapshot_id'].nunique()
data4 = data.groupby('snapshot_id')
data4.dtypes
data2.snapshot_id

xx = np.arange(0,5, 0.001)
yy = np.cos(xx)
plt.plot(xx,yy)
plt.show()
zz = np.vstack((xx,yy))
np.sum(zz)

from scipy import stats
stats.describe(zz)

uaa = pd.read_csv("../uaa_index/data/Data815/flood_result.csv")
os.getcwd()

uaa1 = uaa.sample(n = 20)
uaa2 = uaa.sample(n = 20)

drought = pd.read_csv("../Scripts/hazard probabilities.csv")
drought.describe()
