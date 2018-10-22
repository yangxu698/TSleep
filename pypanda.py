import os
import pandas as pd
import numpy  as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

## import pyarrow.parquet as pq
os.chdir("/home/yang/Lucid/dataset.parquet/")

data_raw = pq.read_table('part-00000-tid-6056519413201330783-119422a4-5c84-4461-ba64-4a5e2a1d4acd-29-c000.snappy.parquet').to_pandas()
data_raw.info()

selector = VarianceThreshold(threshold = 0.2)
## remove column with zero <= 0.2, which means variance <= 0.04 ##
data0 = data_raw.loc[:, data_raw.std() > 0.2]
data0.info()
data_raw['label'].value_counts()
data0 = pd.concat([data_raw['label'], data0], axis = 1)
data0.describe()

## column_scale = data0.std().sort_values(ascending=False)[0:15].index
scaler = MinMaxScaler()
data0.iloc[:,1:] = (scaler.fit_transform(data0.iloc[:,1:]))
data0.describe()

## pd.read_parquet('part-00000-tid-6056519413201330783-119422a4-5c84-4461-ba64-4a5e2a1d4acd-29-c000.snappy.parquet', engine='fastparquet')

data1 = pd.concat([data0[data0.label == 0].sample(n = 70), data0[data0.label == 1]])
data1.describe()
data1.std().value_counts()
data1.corr()

pca = PCA(n_components=0.95)
pca.fit(data1.iloc[:,1:])

data_predictor_new = pca.fit_transform(data0.iloc[:,1:])
data_predictor_new = pd.DataFrame(data_predictor_new)
data2 = pd.concat([data0['label'], data_predictor_new], axis = 1)
data2.info()
data2.describe()
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

data3 = pd.concat([data2[data2.label == 0].sample(n = 100), data2[data2.label == 1]])
data3.info()

clf = SVC(gamma = 'auto')
clf2 = SVC(gamma = 'auto')
predictors = data2.iloc[:,1:]
true_class = data2.iloc[:,0]
from imblearn.over_sampling import SMOTE, ADASYN
sm = SMOTE(random_state=42)
X_resample, Y_resample = sm.fit_sample(predictors, true_class)
X_ADsample, Y_ADsample = ADASYN().fit_sample(predictors, true_class)
len(Y_ADsample)
Y_ADsample[10007]
Y_resample[4999]
svm_weights = np.concatenate((np.repeat(1,5000), np.repeat(0.80,5000)), axis = 0)
clf.fit(predictors, true_class), sample_weight= svm_weights)
clf.fit(X_resample, Y_resample, sample_weight= svm_weights)
clf2.fit(X_ADsample, Y_ADsample)##  sample_weight= svm_weights)
y_pred = clf.predict(predictors)

from sklearn.metrics import classification_report, confusion_matrix
X_test = data2.iloc[:,1:]
Y_test = data2.iloc[:,0]
Y_pred = clf.predict(X_test)

print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
