import json
import random
import numpy as np


def getProzx(xn, muk, covk):
    covkMaxtrix = np.matrix(covk).reshape(2, 2)
    mukMaxtrix = np.matrix(muk).reshape(2, 1)

    xnMaxtrix = np.matrix(xn).reshape(2, 1)

    temp1 = np.transpose(xnMaxtrix - mukMaxtrix)
    temp2 = covkMaxtrix.I
    temp3 = np.dot(temp1, temp2)
    tempfinal = np.dot(temp3, xnMaxtrix - mukMaxtrix)
    numone = tempfinal[0, 0]
    det = np.linalg.det(np.multiply(covkMaxtrix, 2 * np.pi))
    if det < 0:
        det = -det
    gamank = np.exp(-0.5 * numone) / (np.sqrt(det))
    return gamank

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###

    # Run 100 iterations of EM updates
    for t in range(100):

        ###get every gamank shape is (600,3)
        totalsavedgamank = []
        for n in range(len(X)):
            xn = np.array(X[n])
            temppro = []
            for k in range(K):
                everyprozkx = getProzx(xn, mu[k], cov[k])
                temppro.append(everyprozkx)

            onenlist = []
            sumprox = 0
            for i in range(len(temppro)):
                sumprox += temppro[i] * pi[i]
                onenlist.append(temppro[i] * pi[i])

            onenlist = onenlist / sumprox
            totalsavedgamank.append(onenlist)


        ###updata the pi
        totalomega = 0
        sumomega = []
        for k in range(K):
            oneomega = 0

            for n in range(len(X)):
                oneomega += totalsavedgamank[n][k]

            totalomega += oneomega
            sumomega.append(oneomega)

        sumomega = sumomega / totalomega

        for k in range(K):
            pi[k] = sumomega[k]

        ###update the mean
        totalnparray = np.array(totalsavedgamank)
        totalnparrayofXdata = np.array(X)
        for k in range(K):

            mu0 = 0
            mu1 = 0
            sumhehe = 0

            for n in range(len(X)):
                mu0 += totalsavedgamank[n][k] * X[n][0]
                mu1 += totalsavedgamank[n][k] * X[n][1]
                sumhehe += totalsavedgamank[n][k]

            templist =[]
            templist.append(mu0/sumhehe)
            templist.append(mu1/sumhehe)

            mu[k] = templist


        ###update the var
        for k in range(K):
            sig0 = 0
            sig1 = 0
            sig2 = 0
            sig3 = 0
            sumhehe = 0

            for n in range(len(X)):
                temp0 = X[n][0] - mu[k][0]
                temp1 = X[n][1] - mu[k][1]
                sig0 += totalsavedgamank[n][k] * temp0 * temp0
                sig1 += totalsavedgamank[n][k] * temp0 * temp1
                sig2 += totalsavedgamank[n][k] * temp1 * temp0
                sig3 += totalsavedgamank[n][k] * temp1 * temp1
                sumhehe += totalsavedgamank[n][k]

            templist = []
            templist.append(sig0 / sumhehe)
            templist.append(sig1 / sumhehe)
            templist.append(sig2 / sumhehe)
            templist.append(sig3 / sumhehe)

            cov[k] = templist

    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()