import pandas as pd

file_name = "submissions/resultsLogisticRegressionCVfc6fc7.csv"

df = pd.read_csv(file_name, sep=',', header=None, skiprows=1)

df[df.columns[1:]] += 0.01

for i in range(1, 11):
    df.loc[df[i] > 1, i] = 1

df.columns = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

df.to_csv(file_name + '.added', index=False, header=True)
