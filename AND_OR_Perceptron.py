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

def Train(X,Y,W_1,W_2,b, lr , LOSS):
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
			Yhat = Sigmoid(W_1*x[0]+W_2*x[1]+b)
			loss = Criterion(Yhat, Y[i])
			L.append(loss)

			# Derivatives using chain rule (Back-propagation)
			dW_1 = d_Criterion(Yhat , Y[i])*d_Sigmoid(Yhat)*x[0]
			dW_2 = d_Criterion(Yhat , Y[i])*d_Sigmoid(Yhat)*x[1]
			db = d_Criterion(Yhat , Y[i])*d_Sigmoid(Yhat)

			# Updating
			W_1 = W_1 - lr*dW_1
			W_2 = W_2 - lr*dW_2
			b = b - lr*db

		LOSS.append(np.mean(L))

	return W_1, W_2,b, LOSS



"""#input for AND
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,0,0,1])
"""

#input for OR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,1])



"""#input for NAND
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([1,1,1,0])
"""


"""#input for NOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([1,0,0,0])
"""

"""#input for XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,0])
"""


# weights initialization
W_1 = np.random.uniform(-0.01,0.01, size = (1,))
W_2 = np.random.uniform(-0.01,0.01, size = (1,))
b = np.random.uniform(-0.01,0.01, size = (1,))

# training parameters
EPOCH = 4000
lr = 0.01
LOSS = [] 

W_1, W_2,b, LOSS = Train(X,Y,W_1,W_2,b, lr , LOSS)

print("W1 = " , W_1 , "W2 = ", W_2 , "b = ", b)


# plot geometric data and boundary
plt.figure(figsize = (3,3))
plt.grid()

"""# AND
plt.title("AND GATE")
plt.scatter(0,0, c= 'r', label = "class 0")
plt.scatter(0,1, c= 'r', label = "class 0")
plt.scatter(1,0, c= 'r', label = "class 0")
plt.scatter(1,1, c= 'b', label = "class 1")
"""

# OR
plt.title("OR GATE")
plt.scatter(0,0, c= 'r', label = "class 0")
plt.scatter(0,1, c= 'b', label = "class 1")
plt.scatter(1,0, c= 'b', label = "class 1")
plt.scatter(1,1, c= 'b', label = "class 1")


"""# NAND
plt.title("NAND GATE")
plt.scatter(0,0, c= 'b', label = "class 1")
plt.scatter(0,1, c= 'b', label = "class 1")
plt.scatter(1,0, c= 'b', label = "class 1")
plt.scatter(1,1, c= 'r', label = "class 0")
"""

"""# NOR
plt.title("NOR GATE")
plt.scatter(0,0, c= 'b', label = "class 1")
plt.scatter(0,1, c= 'r', label = "class 0")
plt.scatter(1,0, c= 'r', label = "class 0")
plt.scatter(1,1, c= 'r', label = "class 0")
"""

"""# XOR
plt.title("NOR GATE")
plt.scatter(0,0, c= 'b', label = "class 0")
plt.scatter(0,1, c= 'r', label = "class 1")
plt.scatter(1,0, c= 'r', label = "class 1")
plt.scatter(1,1, c= 'r', label = "class 0")
"""


X_1 = np.arange(-1,2, 0.1)
X_2 = -(W_1/W_2)*X_1-(b/W_2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim((-1,2))
plt.ylim((-1,2))
plt.plot(X_2, X_1, c="g")

# plot loss
plt.figure()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("training loss")
plt.plot(LOSS , c = 'g')
plt.show()

