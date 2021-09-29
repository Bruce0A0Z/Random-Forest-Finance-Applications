

import numpy as np;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import kneighbors_graph, NearestNeighbors;
import scipy;
import scipy.sparse as sparse;
import matplotlib.pyplot as plt;

plt.style.use('seaborn-deep');


#PARAMETER SET UP: 
T = 1./12; #Option Expiry
m = 2; #Amount of time we can exercise the option
K = 264; #Strike Price
r = 0.01; #RFR
S0 = 263.24; #Today's (or Initial) Stock Price
sigma = 0.2; #Volatility
N = 10000; #Number of forward simulation used for regression
Q = 0; #Dividend


#FORWARD SIMULATION:
dt = T/m;
S = np.zeros((m+1,N)); S[0] = S0;
dB = np.zeros((m+1,N));
for t in range(1,m+1):
    x = np.random.standard_normal(size = N);
    S[t] = S[t-1]*np.exp((r - sigma**2/2)*dt + sigma*np.sqrt(dt)*x);
    dB[t] = np.sqrt(dt)*x;
    
Y = np.maximum(S[-1] - K,0);
X = S[1];
X = X[:,None];

#Plot
f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,7));
plt.tight_layout(pad = 0.4, w_pad = 6, h_pad = 1.0)

ax1.plot(S[2], Y, 'r.');
ax2.plot(S[1], Y, 'r.'); #Difference in plots caused by the fact that we calculated Y based on S[2]

f.show()


#REGRESSION

#Polynomial Regression (degree = 5):
regl = np.polyfit(S[1], Y, 5);
expected_Y_polynomial = np.polyval(regl, S[1])

#Gaussian kernel regression using only nearest neighbours
NN = kneighbors_graph(X.reshape(N, 1), int(N * 0.01), mode='distance').nonzero()
l = 2
x = NN[0]
y = NN[1]
W = scipy.sparse.lil_matrix((N, N)) # W is the weight matrix, i.e in w(i,j) = exp(-(x_j - x_i)^2 / 2.l^2) / sum_axis_0(exp(-(x_j - x_i)^2 / 2.l^2))
W[x, y] = np.exp(- (X[y] - X[x]) ** 2 / (2 * l ** 2))[0]
W.setdiag(np.ones(N)) # As (i, i) not in NN, we add it manually
sum_weights = W.sum(axis=0) # Sum over each row and update the weight
W = W.dot(scipy.sparse.diags(np.array(1 / sum_weights)[0]))
expected_Y_kernel = W.dot(Y) # E[Y_(t+dt)_j|F_t] = f(X_j) = sum(w(i, j)*Y(i))

#Plot
f2, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(S[1], Y , 'r.')
ax1.plot(S[1], expected_Y_polynomial , 'b.')
ax1.set_title('Polynomial Regression')
ax2.plot(S[1], Y , 'r.')
ax2.plot(S[1], expected_Y_kernel , 'b.')
ax2.set_title('Gaussian Kernel Regression')

#Random Forest Regression
rf = RandomForestRegressor(n_estimators=100,n_jobs=-1)
regl = rf.fit(X,Y)
expected_Y_rf = rf.predict(X)
f3, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.plot(X, Y , 'r.')
ax1.plot(X, expected_Y_rf, 'g.')

