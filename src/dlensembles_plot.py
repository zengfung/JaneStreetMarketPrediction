# plot dlensembles results
# plot results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../results/ensembleresults.csv")
df = df.reset_index()
df = df.sort_values("Test Accuracy", ascending = True)

ind = np.arange(len(df))
width = 0.4

model = np.asarray(df["Model"])
train = np.asarray(df["Train Accuracy"])
test = np.asarray(df["Test Accuracy"])

fig, ax = plt.subplots(figsize = (20,12))
ax.bar(ind - width/2, train, width, color = "yellow", edgecolor = "black")
ax.bar(ind + width/2, test, width, color = "red", edgecolor = "black")
ax.legend(["Train Accuracy", "Test Accuracy"], fontsize = 14)
ax.set_ylim(0,1.05)
ax.set_ylabel("Accuracy", fontsize = 14)
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 14)
ax.set_xticks(ind)
ax.set_xlabel("Models", fontsize = 14)
ax.set_xticklabels(model, rotation = 20, fontsize = 14, ha = "right")
ax.set_title("Comparison of Accuracies between DL models and their Ensembles", fontsize = 18)

for i in range(len(df)):
	ax.text(ind[i] - width/2, train[i] + 0.01, str(round(train[i], 3)),
		ha = "center", va = "bottom", size = 14, rotation = "vertical")
	ax.text(ind[i] + width/2, test[i] + 0.01, str(round(test[i], 3)),
		ha = "center", va = "bottom", size = 14, rotation = "vertical")

# save results
fig.savefig("../results/ensemblechart.png")
fig.show()