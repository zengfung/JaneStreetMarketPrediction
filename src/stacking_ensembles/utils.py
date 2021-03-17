import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV
   
# load models
def load_keras_models(names):
    all_models = list()
    for filename in names:
        model = load_model(filename)
        all_models.append(model)
        print("Loaded {0}".format(filename))
    return all_models

# transforming inputs
def transform_inputs(train, test, types):
    train_stack = list()
    test_stack = list()
    for input_type in types:
        if input_type == "combine":
            train_stack.append([train])
            test_stack.append([test])
            print("combine-type transform success!")
        elif input_type == "split":
            [w_train, ts_train] = [train[:,0], train[:,1:]]
            [w_test, ts_test] = [test[:,0], test[:,1:]]
            train_stack.append([ts_train, w_train])
            test_stack.append([ts_test, w_test])
            print("split-type transform success!")
        elif input_type == "pca_split":
            [w_train, ts_train] = [train[:,0], train[:,1:]]
            [w_test, ts_test] = [test[:,0], test[:,1:]]
            pca = PCA(n_components = 57).fit(ts_train)
            tsf_train = pca.transform(ts_train)
            tsf_test = pca.transform(ts_test)
            train_stack.append([tsf_train, w_train])
            test_stack.append([tsf_test, w_test])
            print("pca_split-type transform success!")
        else: 
            raise Exception("Invalid transform type!")
    return [train_stack, test_stack]

# prediction accuracies for each model
def individual_prediction(members, x_train, y_train, x_test, y_test, results={}):
    for (i, model) in enumerate(members):
        print("Model {0}:".format(i+1))
        yhat_train = model.predict(x_train[i])
        yhat_train = (yhat_train > 0.5).astype("int")
        train_acc = accuracy_score(y_train, yhat_train)
        print("Train accuracy:", train_acc)
        yhat_test = model.predict(x_test[i])
        yhat_test = (yhat_test > 0.5).astype("int")
        test_acc = accuracy_score(y_test, yhat_test)
        print("Test accuracy:", test_acc)
        results["Model {0}".format(i+1)] = {"Train": train_acc, "Test": test_acc}
    return results

# obtain outputs from deep learning models
def get_ensemble_input(members, x, n):
	k = len(members)
	result = np.zeros((n,k))
	for (i, model) in enumerate(members):
		yhat = model.predict(x[i])
		result[:,i] = yhat.reshape((-1,))
	return result

# fit ensemble model
def fit_ensemble(members, x, y, n, model_type, params):
	# get ensemble model input
    ensemble_input = get_ensemble_input(members, x, n)
    clf = GridSearchCV(model_type, params, refit = True, return_train_score = True)
    clf.fit(ensemble_input, y.ravel())
    return clf

# make prediction using ensemble model
def ensemble_predict(members, model, x, n):
	# get ensemble model input
	ensemble_input = get_ensemble_input(members, x, n)
	# make prediction
	yhat = model.predict(ensemble_input)
	return yhat

