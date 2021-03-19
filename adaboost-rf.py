
# decision tree model
import pandas as pd
x = pd.read_csv("./input_data.csv").to_numpy()
resp = pd.read_csv("./output_data.csv").to_numpy()

##
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = np.ravel(y)

##
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)
print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# add gridsearch
model = RandomForestClassifier()
ABC = AdaBoostClassifier(base_estimator = model)
params = {"base_estimator__criterion" : ["gini", "entropy"],
          "base_estimator__min_samples_split" : [2, 20, 200],
          "n_estimators" : [50, 100, 500]}

# fit model
clf = GridSearchCV(ABC, params)
clf.fit(x_train, y_train)

# grid search results
results= pd.DataFrame.from_dict(clf.cv_results_)
print(results)
results.to_csv("./boostforest.csv")  # save as csv file
print("Best model score:", clf.best_score_)
print(clf.best_params_)

# predictions on training set
yhat_train = clf.predict(x_train)
print("Train MSE:", accuracy_score(y_train, yhat_train))

# predictions on test set
yhat_test = clf.predict(x_test)
print("Test MSE:", accuracy_score(y_test, yhat_test))






