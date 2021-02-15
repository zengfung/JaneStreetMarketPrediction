# code to clean out data from train.csv

import pandas as pd

df = pd.read_csv("./jane-street-market-prediction/train.csv")

# drop rows with NaN values
df.dropna(axis = 0, inplace = True)

# create data frame for inputs (X)
ts = df.iloc[:, 7:137]
w = df.iloc[:, 1]
x = pd.concat([w.reset_index(drop=True), ts.reset_index(drop=True)], axis = 1)

# create data frame for ouputs (y)
y = df.iloc[:, 2:7]

x.to_csv("./dataset/input_data.csv", index=False)
y.to_csv("./dataset/output_data.csv", index=False)
