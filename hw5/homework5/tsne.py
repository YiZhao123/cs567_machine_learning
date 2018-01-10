import numpy as np
import pylab as plt
import sys




def pca(X = np.array([]), initial_dims = 50):
    """
    Runs PCA on the N x D array X in order to reduce its dimensionality to no_dims dimensions.

    Inputs:
    - X: An array with shape N x D where N is the number of examples and D is the
         dimensionality of original data.
    - initial_dims: A scalar indicates the output dimension of examples after performing PCA.

    Returns:
    - Y: An array with shape N x initial_dims where N is the number of examples and initial_dims is the
         dimensionality of output examples. intial_dims should be smaller than D, which is the
         dimensionality of original examples.
    """

    print "Preprocessing the data using PCA..."

    Y = np.zeros((X.shape[0], initial_dims))

    # start your code here
    # num_data, dim = X.shape

    mean_vals = X.mean(axis=0)
    mean_removed = X - mean_vals

    cov_mat = np.cov(mean_removed, rowvar=0)

    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_index = np.argsort(eig_vals)

    eig_val_index = eig_val_index[:-(initial_dims + 1): -1]
    reg_eig_vects = eig_vects[:, eig_val_index]

    Y = np.matmul(mean_removed, reg_eig_vects)
    Y = np.array(Y)

    return Y

def compute_Q(Y):
    """
    Compute pairwise affinities of Y.

    Inputs:
    - Y: An array with shape N x no_dims, where N is the number of examples and no_dims
         is the dimensionality of lower dimensional data Y.

    Returns:
    - Q: An array with shape N x N, where N is the number of examples. Q[i][j] is the
         joint probability of Y[i] and Y[j] defined in equation *.
    - Y_sim: An array with the same shape with Q. Y_sim[i][j] is the numerator of Q[i][j] 
         in equation *.
    """
    # print ("############################## compute_Q ###############################")

    Y_sim = np.zeros((Y.shape[0], Y.shape[0]))
    Q = np.zeros((Y.shape[0], Y.shape[0]))

    # start your code here
    ydata = Y
    sum_ydata = np.sum(np.square(Y),axis=1)
    sum_ydata = sum_ydata.reshape(2500,1)
    sum_ydataT = np.transpose(sum_ydata)
    ydataT = np.transpose(ydata)
    tempdata1 = -2*np.dot(ydata, ydataT)
    tempdata2 = sum_ydataT + tempdata1

    tempdata3= sum_ydata + tempdata2
    tempdata3 = tempdata3 + 1

    Y_sim = np.reciprocal(tempdata3)
    np.fill_diagonal(Y_sim, 0)

    totalsum = np.sum(Y_sim)
    Q = Y_sim / totalsum

    """total = 0
    N = len(Y)
    for i in range(N):
        for j in range(N):
            if i == j:
                Y_sim[i][j] = 0
                continue
            else:
                one = Y[i]
                two = Y[j]
                Y_sim[i][j] = 1.0 / (1 + np.sum(np.square(one - two)))
                total += Y_sim[i][j]

    for i in range(N):
        for j in range(N):
            Q[i][j] = Y_sim[i][j] / total"""

    return Q, Y_sim

def compute_gradient(P, Q, Y_sim, Y, dY, no_dims):
    """
    Compute the gradients.
    
    Inputs:
    - P: An array with shape N x N, where N is the number of examples. P[i][j] is the 
         joint probability of Y[i] and Y[j] defined in equation *.
    - Q: An array with shape N x N, where N is the number of examples. Q[i][j] is the
         joint probability of Y[i] and Y[j] defined in equation *.
    - Y_sim: An array with the same shape with Q. Y_sim[i][j] is the nominator of Q[i][j] 
         in equation *.
    - Y: An array with shape N x no_dims, where N is the number of examples and no_dims
         is the dimensionality of lower dimensional data Y.
    - dY: An array with same shape of Y, where dY[i][j] is the gradient of Y[i][j].
    - no_dims: A scalar indicates the output dimension of examples after performing t-SNE.
    
    Returns:
    - dY: An array with same shape of Y, where dY[i][j] is the gradient of Y[i][j].
    """

    # print ("############################## compute_gradient ###############################")
    part = P - Q
    dY = np.zeros(Y.shape)
    # start your code here
    tempart = np.multiply(part, Y_sim)
    onearr = np.sum(tempart,axis = 0)
    diagmatrix = np.diag(onearr)
    matrixnew = diagmatrix - tempart
    dY = 4*np.dot(matrixnew, Y)

    """for i in range(len(dY)):
        temp = np.zeros(len(dY[i]))
        for j in range(len(Y)):
            temp += tempart[i][j] * (Y[i] - Y[j])
        dY[i] = 4*temp"""

    return dY

def Hbeta(D = np.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
	return P;

def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
    """
    Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float."
        return -1
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer."
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real

    if sys.argv[1] == 'pca':
        np.save('pca.npy',X[0:3,:])
        exit()
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.load('Y_init.npy')
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)                            
    P = P * 4                                    # early exaggeration
    P = np.maximum(P, 1e-12)                     # joint probability matrix

    # ######################################## experiment ##############################################
    # Q, Y_sim = compute_Q(Y)
    # Q = np.maximum(Q, 1e-12)
    # dY = compute_gradient(P, Q, Y_sim, Y, dY, no_dims)
    # ######################################## experiment is over ##############################################

    # Run iterations
    for iter in range(max_iter):
        print "in iteration"
        # Compute pairwise affinities
        Q, Y_sim = compute_Q(Y)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        dY = compute_gradient(P, Q, Y_sim, Y, dY, no_dims)


        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print "Iteration ", (iter + 1), ": error is ", C

        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    return Y


if __name__ == "__main__":
    print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
    print "Running example on 2,500 MNIST digits..."
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 20.0)
    if sys.argv[1] == 'output':
        np.save('Y.npy', Y)
    plt.scatter(Y[:,0], Y[:,1], 20, labels)
    plt.show()
