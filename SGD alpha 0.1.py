
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

def h(X, Theta, Bias):
	for i in np.arange(0, X.shape[0], Bias):
		yield (X[i:i + Bias], Theta[i:i + Bias])

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=60,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.1,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

# diambil samples dari data iris 100 data pertama
(X, Theta) = make_blobs(n_samples=100, n_features=2, centers=2,
	cluster_std=2.5, random_state=95)

X = np.c_[np.ones((X.shape[0])), X]

print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

lossHistory = []

for epoch in np.arange(0, args["epochs"]):
	epochLoss = []
	for (batchX, batchY) in h(X, Theta, args["epochs"]):
		preds = sigmoid_activation(batchX.dot(W))
		error = preds - batchY
		loss = np.sum(error ** 2)
		epochLoss.append(loss)

		gradient = batchX.T.dot(error) / batchX.shape[0]
		W += -args["alpha"] * gradient
	lossHistory.append(np.average(epochLoss))

Y = (-W[0] - (W[1] * X)) / W[2]

plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=Theta)
plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Error")
plt.show()