# plot dlensembles results
# plot results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../results/ensembleresults.csv")
df = df.reset_index()
# convert results to percentage
df["Train Accuracy"] = df["Train Accuracy"] * 100
df["Test Accuracy"] = df["Test Accuracy"] * 100

ind = np.arange(len(df))
width = 0.4

model = np.asarray(df["Model"])
train = np.asarray(df["Train Accuracy"])
test = np.asarray(df["Test Accuracy"])

fig, ax = plt.subplots(figsize = (20,12))
ax.bar(ind - width/2, train, width, color = "red", edgecolor = "black")
ax.bar(ind + width/2, test, width, color = "green", edgecolor = "black")
# ax.legend(["Train Accuracy", "Test Accuracy"], fontsize = 14)
ax.set_ylim(0,100)
ax.set_ylabel("Accuracy", fontsize = 14)
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize = 14)
ax.set_xticks(ind)
ax.set_xlim(-0.5, len(df)-0.5)
ax.set_xlabel("Models", fontsize = 14)
ax.set_xticklabels(model, rotation = 20, fontsize = 14, ha = "right")
ax.set_title("Comparison of Accuracies between DL models and their Ensembles", fontsize = 18)

for i in range(len(df)):
	ax.text(ind[i] - width/2, train[i] - 0.01, str(round(train[i], 2)) + "%",
		ha = "center", va = "top", size = 17, rotation = "vertical", color = "white")
	ax.text(ind[i] + width/2, test[i] - 0.01, str(round(test[i], 2)) + "%",
		ha = "center", va = "top", size = 17, rotation = "vertical", color = "white")
    
# add horizontal line representing deeplearning models average accuracy
# deeplearning models average accuracy
dlacc_train = 73.82
dlacc_test = 63.76
x = np.arange(-1, len(df)+1)
# add line
ax.plot(x, [dlacc_test] * (len(df)+2), "b--")
ax.plot(x, [dlacc_train] * (len(df)+2), "k-.")
ax.legend(["DeepLearning Avg. Train Acc.", "DeepLearning Avg. Test Acc.", "Train Accuracy", "Test Accuracy"], fontsize = 14)

# save results
fig.savefig("../../results/ensemblechart.png")
fig.show()