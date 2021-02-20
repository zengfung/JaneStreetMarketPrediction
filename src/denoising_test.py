import pandas as pd
from utils import denoise_train_signals
import time

x = pd.read_csv("../dataset/input_data.csv", nrows = 10000).to_numpy()
signals = x[:,2:130]

start = time.time()
signals, t = denoise_train_signals(signals, wt = "haar")
end = time.time()

print("Elapsed time:", end - start)
print("t:", t)