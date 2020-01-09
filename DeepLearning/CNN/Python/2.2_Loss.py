import numpy as np

D = 784
K = 10
N = 128

scores = np.random.randn(N, K)
y = np.random.randint(K, size = N)

exp_scores = np.exp(scores)