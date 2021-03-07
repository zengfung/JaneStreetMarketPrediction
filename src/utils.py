import pywt
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# select best threshold value
def best_threshold_value(coef, makeplot = False):
    # sort coefficients by magnitude in descending order
    x = -np.sort(-np.absolute(coef))

    # compute the relative error curve
    r = orth2relerror(coef)

    # shift the data points
    x = np.append(x, 0)
    r = np.append(r[0], r)

    # reorder the data points
    xmax = np.amax(x)
    ymax = np.amax(r)
    x = np.flip(x) / xmax
    y = np.flip(r) / ymax

    # compute elbow method
    ix, A, v = [], [], []
    for j in range(3):
        if j == 0:
            ix_j, A_j, v_j = find_elbow(x, y)
        else:
            ix_j, A_j, v_j = find_elbow(x[0:(ix[j-1]+1)], y[0:(ix[j-1]+1)])
        ix.append(ix_j)
        A.append(A_j)
        v.append(v_j) 

    if makeplot:
        threshold_vs_relerror_plot(x*xmax, y*ymax, ix, A, v)

    return x[ix[-1]] * xmax


# orth2relerror
def orth2relerror(orth):
    # sort the coefficients in descending order
    orth = -np.sort(-np.power(orth, 2))

    # compute the relative errors
    return np.power(np.absolute(np.sum(orth) - np.cumsum(orth)), 0.5) / np.power(np.sum(orth), 0.5)


# find elbow points
def find_elbow(x, y):
    # a unit vector pointing from (x1, y1) to (xN, yN)
    v = np.array([x[-1] - x[0], y[-1] - y[0]])      
    v = v / np.linalg.norm(v, 2)

    # subtract (x1, y1) from the coordinates
    xy = np.array([x - x[0], y - y[0]])

    # hypothenuse
    H = np.power(np.sum(np.power(xy, 2), axis = 0), 0.5)

    # adjacent
    A = np.dot(v, xy)                               

    # opposite
    O = np.power(np.absolute(np.power(H, 2) - np.power(A, 2)), 0.5)

    # return the largest distance
    return (np.argmax(O), A, v)


# threshold_vs_relerror_plot
def threshold_vs_relerror_plot(x, y, ix, A, v):
    # rescale x and y values
    xmax = np.amax(x)
    ymax = np.amax(y)

    # relative error line
    fig, ax = plt.subplots()
    ax.plot(x, y, color = "black", linestyle = "-", linewidth = 2)
    ax.set_xlim([0, 1.004*xmax])
    ax.set_ylim([0, 1.004*ymax])
    ax.set_xlabel("Threshold value")
    ax.set_ylabel("Relative errors")

    # first elbow point
    ax.plot([x[0], x[-1]], [y[0], y[-1]], color = "blue", linestyle = "-", linewidth = 2)                   # diagonal line
    dropto = np.array([x[0], y[0]]) + A[0][ix[0]] * (v[0] * np.array([xmax, ymax])) 
    ax.plot([x[ix[0]], dropto[0]], [y[ix[0]], dropto[1]], color = "blue", linestyle = "-", linewidth = 2)   # perpendicular line
    ax.scatter(x[ix[0]], y[ix[0]], color = "blue", linewidth = 2)                                           # higlight point

    # second elbow point
    ax.plot([x[0], x[ix[0]]], [y[0], y[ix[0]]], color = "green", linestyle = "-", linewidth = 2)            # diagonal line
    dropto = np.array([x[0], y[0]]) + A[1][ix[1]] * (v[1] * np.array([xmax, ymax])) 
    ax.plot([x[ix[1]], dropto[0]], [y[ix[1]], dropto[1]], color = "green", linestyle = "-", linewidth = 2)  # perpendicular line
    ax.scatter(x[ix[1]], y[ix[1]], color = "green", linewidth =2)                                           # higlight point

    # third elbow point
    ax.plot([x[0], x[ix[1]]], [y[0], y[ix[1]]], color = "red", linestyle = "-", linewidth = 2)              # diagonal line
    dropto = np.array([x[0], y[0]]) + A[2][ix[2]] * (v[2] * np.array([xmax, ymax])) 
    ax.plot([x[ix[2]], dropto[0]], [y[ix[2]], dropto[1]], color = "red", linestyle = "-", linewidth = 2)    # perpendicular line
    ax.scatter(x[ix[2]], y[ix[2]], color = "red", linewidth = 2)                                            # higlight point

    fig.show()
    return


# denoise training signals
def denoise_train_signals(x, wt):
    (N, n) = x.shape
    
    # discrete wavelet decomposition + obtain mean of best threshold value
    coefs, ts = [], np.zeros(N)
    for i in range(N):
        coefs.append(pywt.wavedec(x[i,:], wt, mode = "periodization", level = pywt.dwt_max_level(n, wt)))
        coef_arr,_ = pywt.coeffs_to_array(coefs[i])
        ts[i] = best_threshold_value(coef_arr)
    t = np.mean(ts)

    # threshold coefs using t + reconstruct denoised signals
    x_denoised = np.zeros(x.shape)
    for i in range(N):
        coefs[i][1:] = (pywt.threshold(j, value = t, mode = "soft", substitute = 0) for j in coefs[i][1:])  # thresholding not done on lowest frequency node
        x_denoised[i,:] = pywt.waverec(coefs[i], wt, mode = "periodization")
    
    return x_denoised, t


# denoise test signals
def denoise_test_signals(x, wt, t):
    (N,n) = x.shape

    # discrete wavelet decomposition + thresholding + reconstruction
    x_denoised = np.zeros(x.shape)
    for i in range(N):
        coef = pywt.wavedec(x[i,:], wt, mode = "periodization", level = pywt.dwt_max_level(n, wt))
        coef[1:] = (pywt.threshold(j, value = t, mode = "soft", substitute = 0) for j in coef[1:])          # thresholding not done on lowest frequency node
        x_denoised[i,:] = pywt.waverec(coef, wt, mode = "periodization")

    return x_denoised


# denoise all signals
def denoise_signals(x_train, x_test, wt = "haar"):
    x_train_denoised, t = denoise_train_signals(x_train, wt)
    x_test_denoised = denoise_test_signals(x_test, wt, t)

    return x_train_denoised, x_test_denoised, t

# DELETE!
# turn the weights into an extra channel
def weights2channels(x, weights):
    (N, n) = x.shape
    
    x_new = np.zeros((N,n,2))
    for i in range(N):
        x_new[i,:,0] = x[i,:]
        x_new[i,:,1] = weights[i]
    
    return x_new

# calculates entropy of a signal
def calculate_entropy(list_values):
    counter_values = collections.Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    et = entropy(probabilities)
    return et

# get summary statistics of the signal
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.sqrt(np.nanmean(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

# calculate crossings in the signal
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

# get features of signal
def get_features(list_values):
    et = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [et] + crossings + statistics

# gets features for all signals by calling get_features()
def get_signals_features(x, wt):
    (N, n) = x.shape
    max_level = pywt.dwt_max_level(n, wt)
    features = np.zeros((N, (max_level + 1) * 12))
    for i in range(N):
        list_coef = pywt.wavedec(x[i,:], wt, mode = "periodization", level = max_level)
        for (j, coef) in enumerate(list_coef):
            features[i, (j*12):((j+1)*12)] = np.asarray(get_features(coef))
    return features

# obtains scalogram of a signal
def get_scalogram(x, wt = "morl"):
    n = x.shape[0]
    scales = range(1, n+1)
    coef, freq = pywt.cwt(x, scales, wt)
    coef_ = coef[:,:130]
    return np.power(coef_, 2)

# obtains scalogram for all signals
def get_signals_scalogram(x, wt = "morl"):
    N, n = x.shape
    scalograms = np.zeros((N, n, n))
    for i in range(N):
        scalograms[i,:,:] = get_scalogram(x[i,:], wt)
    return scalograms

# plot the scalogram
def plot_scalogram(x, y, wt = "morl", count = 9):
    n = x.shape[1]
    scales = range(1, n+1)
    i = 0
    counter_1, counter_0 = 0, 0
    fig1, ax1 = plt.subplots(3,3)
    fig1.suptitle("Energy map for Action = 1")
    fig0, ax0 = plt.subplots(3,3)
    fig0.suptitle("Energy map for Action = 0")
    
    while True:
        coef,_ = pywt.cwt(x[i,:], scales, wt)
        pw = np.power(coef, 2)
        if counter_1 < 9 and y[i,0] == 1:
            row = counter_1 // 3
            col = counter_1 % 3
            ax1[row, col].contourf(pw)
            counter_1 += 1
        elif counter_0 < 9 and y[i,0] == 0:
            col = counter_0 % 3
            row = counter_0 // 3
            ax0[row, col].contourf(pw)
            counter_0 += 1
        elif counter_1 >= 9 and counter_0 >= 9:
            break
        i += 1
    
    fig1.savefig("../results/energymaps1.png")
    fig0.savefig("../results/energymaps0.png")
    fig1.show()
    fig0.show()
   
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