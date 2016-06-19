import pandas as pd
import numpy as np

file_name = "submissions/resultsLogisticRegressionWithCustomScorefc6fc7.csv"

df = pd.read_csv(file_name, sep=',', header=None, skiprows=1)

zero = 45e-6

a = df.values[:, 1:]

df.loc[:, 1:] = np.maximum(np.minimum(a, 1 - zero), zero)

df.columns = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

df.to_csv(file_name + '.threshold', index=False, header=True, float_format="%e")