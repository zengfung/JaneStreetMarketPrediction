import pywt
import numpy as np
import matplotlib.pyplot as plt


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
    for j in range(4):
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

    # fourth elbow point
    ax.plot([x[0], x[ix[2]]], [y[0], y[ix[2]]], color = "cyan", linestyle = "-", linewidth = 2)              # diagonal line
    dropto = np.array([x[0], y[0]]) + A[3][ix[3]] * (v[3] * np.array([xmax, ymax])) 
    ax.plot([x[ix[3]], dropto[0]], [y[ix[3]], dropto[1]], color = "cyan", linestyle = "-", linewidth = 2)    # perpendicular line
    ax.scatter(x[ix[3]], y[ix[3]], color = "cyan", linewidth = 2)                                            # higlight point

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

# turn the weights into an extra channel
def weights2channels(x, weights):
    (N, n) = x.shape
    
    x_new = np.zeros((N,n,2))
    for i in range(N):
        x_new[i,:,0] = x[i,:]
        x_new[i,:,1] = weights[i]
    
    return x_new