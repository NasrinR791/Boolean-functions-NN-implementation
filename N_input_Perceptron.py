from itertools import product
import  numpy as np
import matplotlib.pyplot as plt


#Activation function, we do not use step function becauze it is not differentiable
def Sigmoid(z):
	return 1/(1+np.exp(-z))

# dSigmoid/dz
def d_Sigmoid(z):
	return z*(1-z)


# Binary Ceoss Entropy loss
def Criterion(Yhat, Y):
	return -Y*np.log(Yhat)-(1-Y)*np.log(1-Yhat)

# dCriterion/dYhat
def d_Criterion(Yhat , Y):
	return -(Y/Yhat)+((1-Y)/(1-Yhat))

def Train(X,Y,W,b, lr , LOSS):
	for epoch in range(EPOCH):
		# Shuffle the training data
		random_index = np.arange(X.shape[0])
		np.random.shuffle(random_index)


		# loss list
		L = []

		# loop over data
		for i in random_index:
			x = X[i]

			# Forward pass
			Yhat = Sigmoid(np.inner(x,W)+b)
			loss = Criterion(Yhat, Y[i])
			L.append(loss)

			# Derivatives using chain rule (Back-propagation)

			dW = d_Criterion(Yhat , Y[i])*d_Sigmoid(Yhat)*x
			print(x.shape, dW.shape)
			db = d_Criterion(Yhat , Y[i])*d_Sigmoid(Yhat)

			# Updating
			W = W - lr*dW
			b = b - lr*db

		LOSS.append(np.mean(L))

	return W,b, LOSS


n = 4
X = list(product(range(2), repeat = n))
y =[]


"""# AND
for i in range(len(X)):
	if sum(X[i]) == n:
		y.append(1)
	else:
		y.append(0)
"""

# OR
for i in range(len(X)):
	if sum(X[i]) == 0:
		y.append(0)
	else:
		y.append(1)

X = np.array(X)
Y = np.array(y)

print(X)
print(Y)

W = np.random.uniform(-0.01,0.01, size = (n,))
b = np.random.uniform(-0.01,0.01, size = (1,))

EPOCH = 10000
lr = 0.01
LOSS = [] 

W,b, LOSS = Train(X,Y,W,b, lr , LOSS)

print("W = " , W , "b = ", b)
# plot loss
plt.figure()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("training loss")
plt.plot(LOSS , c = 'g')
plt.show()



