import os
import pandas as pd
import numpy as np
data = pd.read_csv("../Tesserae/derived_features_9_23.csv")
data.describe()
data.shape
data1 = data.iloc[:, [0,5,9,10]]
data1.describe()
data2 = data.sample(n = 10)
print(data2)
