import numpy as np
import matplotlib.pyplot as plt


def Sigmoid(x):
	return 1/(1+np.exp(-x))

def dSigmoid(z):
	return z*(1-z)

def Criterion(Y, Yhat):
	return -Y*np.log(Yhat)-(1-Y)*np.log(1-Yhat)

def dCriterion(Y,Yhat):
	return -(Y/Yhat)+((1-Y)/(1-Yhat))

def Forward(X, W_1, W_2, b_1, b_2):
	Z1 = np.dot(X,W_1) +b_1
	A1 = Sigmoid(Z1)
	Z2 = np.dot(A1,W_2)+b_2
	A2 = Sigmoid(Z2)
	cache = Z1,A1,Z2,A2,X, W_1, W_2, b_1, b_2
	return A2 , cache

def Backward(y,cache):
	Z1,A1,Z2,A2,X, W_1, W_2, b_1, b_2 = cache

	dZ2 = dCriterion(y,A2)*dSigmoid(A2)
	dW_2 = A1.T.dot(dZ2)
	db_2 = dZ2
	
	dZ1 = np.multiply(dZ2.dot(W_2.T),np.multiply(A1,1-A1))
	dW_1 = X.T.dot(dZ1)
	db_1 = dZ1
	return dW_1, dW_2, db_1, db_2


def train(X,Y,  W_1, W_2, b_1, b_2, EPOCH , lr , LOSS):
	for epoch in range(EPOCH):
		shuffeled_index = np.arange(X.shape[0])
		np.random.shuffle(shuffeled_index)
		E =[]
		for i in shuffeled_index:
			x = np.mat(X[i])
			y = Y[i]
			yhat , cache = Forward(x, W_1, W_2, b_1, b_2)
			Error = Criterion(y, yhat)
			E.append(Error)
			dW_1, dW_2, db_1, db_2 =  Backward(y,cache)

			# Update parameters
			W_1 = W_1 - lr*dW_1
			W_2 = W_2 - lr*dW_2
			b_1 = b_1 - lr*b_1
			b_2 = b_2 - lr*b_2

		LOSS.append(np.mean(E))
		
	return LOSS, W_1,W_2,b_1,b_2 , yhat


# input 
X = np.array([[0,0], [0,1],[1,0],[1,1]])
Y = np.array([0,1,1,0])


Input_dim = 2
Output_dim = 1
Hidden_dim = 2

# initialization
#W_1 = np.random.uniform(-0.01,0.01,size = (Input_dim,Hidden_dim))
#b_1 = np.random.uniform(-0.01,0.01,size = (1,2))
#W_2 = np.random.uniform(-0.01,0.01,size = (Hidden_dim, Output_dim))
#b_2 = np.random.uniform(-0.01,0.01,size = (1,1))

W_1 = np.random.rand(Input_dim,Hidden_dim)
b_1 = np.random.rand(1,2)
W_2 = np.random.rand(Hidden_dim, Output_dim)
b_2 = np.random.rand(1,1)


EPOCH = 100000
lr = 0.1
LOSS =[]

LOSS, W_1,W_2,b_1,b_2 , yhat = train(X,Y,  W_1, W_2, b_1, b_2, EPOCH , lr , LOSS)

#Y_hat = Sigmoid(np.dot(W_2,Sigmoid(np.dot(W_1,X)+b_1))+b_2)
#print(Y_hat)

plt.figure()
plt.grid()
plt.title("Training Loss for XOR", fontsize = 20)
plt.plot(LOSS, linewidth =3)
plt.xlabel("epochs", fontsize = 18)
plt.ylabel("Loss" , fontsize = 18)
plt.show()